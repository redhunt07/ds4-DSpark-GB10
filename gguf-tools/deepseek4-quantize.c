/*
 * DeepSeek V4 Flash/Pro HF -> GGUF quantizer.
 *
 * This is a plain C, model-specific version of the DS4 quantization pipeline.
 * It deliberately keeps only the pieces needed by the DeepSeek V4 Flash and
 * Pro GGUF recipes used by this repository:
 *
 * - safetensors index/header loading;
 * - FP8 E4M3 + E8M0 dequantization for dense tensors;
 * - packed FP4 + E8M0 dequantization for routed experts;
 * - local Q8_0, Q4_K, Q2_K, and IQ2_XXS quantization;
 * - GGUF metadata/tensor-order synthesis from HF config, or reuse from an
 *   existing template GGUF when explicitly requested.
 *
 * The optional imatrix is the legacy llama.cpp binary .dat format emitted by
 * ds4's collector.  DS4 stores one packed vector per routed tensor, laid out as
 * n_experts consecutive per-expert importance vectors.  When no external
 * imatrix is supplied and IQ2_XXS requires one, this tool falls back to the
 * same synthetic weight-energy heuristic used by the old generator:
 * each column importance is sum(row[column]^2) over the dequantized weight.
 */

#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(_WIN32)
#error "deepseek4-quantize.c currently targets POSIX systems"
#endif

#define DS4_KV_QUANTIZE_IMATRIX_FILE      "quantize.imatrix.file"
#define DS4_KV_QUANTIZE_IMATRIX_DATASET   "quantize.imatrix.dataset"
#define DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES "quantize.imatrix.entries_count"
#define DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS  "quantize.imatrix.chunks_count"
#define DS4_GGUF_DEFAULT_ALIGNMENT 32

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static char *path_join(const char *a, const char *b) {
    const size_t na = strlen(a);
    const size_t nb = strlen(b);
    const bool slash = na && a[na - 1] == '/';
    char *out = xmalloc(na + (slash ? 0 : 1) + nb + 1);
    memcpy(out, a, na);
    size_t pos = na;
    if (!slash) out[pos++] = '/';
    memcpy(out + pos, b, nb + 1);
    return out;
}

static bool str_starts(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool str_ends(const char *s, const char *suffix) {
    const size_t ns = strlen(s);
    const size_t nf = strlen(suffix);
    return ns >= nf && memcmp(s + ns - nf, suffix, nf) == 0;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open", path);
    if (fseeko(fp, 0, SEEK_END) != 0) die_errno("seek", path);
    off_t n = ftello(fp);
    if (n < 0) die_errno("tell", path);
    if (fseeko(fp, 0, SEEK_SET) != 0) die_errno("seek", path);
    char *buf = xmalloc((size_t)n + 1);
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("read", path);
    buf[n] = '\0';
    fclose(fp);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static uint64_t read_u64_le_fp(FILE *fp, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

static uint32_t read_u32_le_fp(FILE *fp, const char *what) {
    uint32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

static int32_t read_i32_fp(FILE *fp, const char *what) {
    int32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

static uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int64_t load_i64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return (int64_t)v;
}

/* =====
 * Minimal JSON tokenizer
 *
 * Safetensors uses ordinary JSON for the model index and per-shard headers.
 * We only need objects, arrays, strings, and primitive numbers; escaped tensor
 * names do not occur in the files produced by Hugging Face, so strings are
 * copied as raw UTF-8 slices after locating the closing quote.
 */

typedef enum {
    JT_OBJECT,
    JT_ARRAY,
    JT_STRING,
    JT_PRIMITIVE,
} json_type;

typedef struct {
    json_type type;
    int start;
    int end;
    int parent;
    int size;
} json_tok;

typedef struct {
    json_tok *v;
    int len;
    int cap;
    const char *js;
    int js_len;
} json_doc;

static int json_add(json_doc *d, json_type type, int start, int end, int parent) {
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        d->v = xrealloc(d->v, (size_t)d->cap * sizeof(d->v[0]));
    }
    int id = d->len++;
    d->v[id] = (json_tok){ .type = type, .start = start, .end = end, .parent = parent, .size = 0 };
    if (parent >= 0) d->v[parent].size++;
    return id;
}

static json_doc json_parse_text(const char *js, size_t len) {
    json_doc d = { .js = js, .js_len = (int)len };
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        unsigned char c = (unsigned char)js[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            parent = json_add(&d, c == '{' ? JT_OBJECT : JT_ARRAY, i, -1, parent);
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0) die("bad JSON: unmatched close");
            d.v[parent].end = i + 1;
            parent = d.v[parent].parent;
            continue;
        }
        if (c == '"') {
            int start = i + 1;
            i++;
            bool esc = false;
            for (; i < (int)len; i++) {
                if (esc) {
                    esc = false;
                } else if (js[i] == '\\') {
                    esc = true;
                } else if (js[i] == '"') {
                    break;
                }
            }
            if (i >= (int)len) die("bad JSON: unterminated string");
            json_add(&d, JT_STRING, start, i, parent);
            continue;
        }
        int start = i;
        while (i < (int)len && !isspace((unsigned char)js[i]) &&
               js[i] != ',' && js[i] != ']' && js[i] != '}') {
            i++;
        }
        json_add(&d, JT_PRIMITIVE, start, i, parent);
        i--;
    }
    if (parent != -1) die("bad JSON: unterminated object/array");
    return d;
}

static void json_free(json_doc *d) {
    free(d->v);
    memset(d, 0, sizeof(*d));
}

static bool json_tok_eq(const json_doc *d, int tok, const char *s) {
    const json_tok *t = &d->v[tok];
    const int n = t->end - t->start;
    return t->type == JT_STRING && (int)strlen(s) == n && memcmp(d->js + t->start, s, (size_t)n) == 0;
}

static char *json_strdup_tok(const json_doc *d, int tok) {
    const json_tok *t = &d->v[tok];
    return xstrndup(d->js + t->start, (size_t)(t->end - t->start));
}

static bool json_is_descendant(const json_doc *d, int tok, int parent) {
    for (int p = d->v[tok].parent; p >= 0; p = d->v[p].parent) {
        if (p == parent) return true;
    }
    return false;
}

static int json_skip(const json_doc *d, int tok) {
    int i = tok + 1;
    while (i < d->len && json_is_descendant(d, i, tok)) i++;
    return i;
}

static int json_obj_get(const json_doc *d, int obj, const char *key) {
    if (obj < 0 || d->v[obj].type != JT_OBJECT) return -1;
    for (int i = obj + 1; i < d->len && d->v[i].parent == obj;) {
        int k = i;
        int v = i + 1;
        if (v >= d->len || d->v[v].parent != obj) return -1;
        if (json_tok_eq(d, k, key)) return v;
        i = json_skip(d, v);
    }
    return -1;
}

static int64_t json_i64(const json_doc *d, int tok) {
    char tmp[64];
    const int n = d->v[tok].end - d->v[tok].start;
    if (n <= 0 || n >= (int)sizeof(tmp)) die("bad JSON integer");
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtoll(tmp, NULL, 10);
}

/* =====
 * Small string hash map
 */

typedef struct {
    char *key;
    int value;
} hslot;

typedef struct {
    hslot *slots;
    int cap;
} hmap;

static uint64_t fnv1a_str(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static void hmap_build(hmap *m, char **keys, int n) {
    int cap = 1;
    while (cap < n * 3) cap <<= 1;
    m->cap = cap ? cap : 2;
    m->slots = xcalloc((size_t)m->cap, sizeof(m->slots[0]));
    for (int i = 0; i < n; i++) {
        uint64_t h = fnv1a_str(keys[i]);
        int p = (int)(h & (uint64_t)(m->cap - 1));
        while (m->slots[p].key) p = (p + 1) & (m->cap - 1);
        m->slots[p].key = keys[i];
        m->slots[p].value = i;
    }
}

static int hmap_get(const hmap *m, const char *key) {
    if (!m->slots) return -1;
    uint64_t h = fnv1a_str(key);
    int p = (int)(h & (uint64_t)(m->cap - 1));
    while (m->slots[p].key) {
        if (strcmp(m->slots[p].key, key) == 0) return m->slots[p].value;
        p = (p + 1) & (m->cap - 1);
    }
    return -1;
}

static void hmap_free(hmap *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

/* =====
 * safetensors database
 */

#define MAX_DIMS 8

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint64_t begin;
    uint64_t end;
} st_info;

typedef struct {
    char *name;
    char *file;
} weight_map_entry;

typedef struct {
    char *name;
    st_info info;
} tensor_entry;

typedef struct {
    char *file;
    char *path;
    uint64_t data_base;
    tensor_entry *tensors;
    int n_tensors;
    int cap_tensors;
    hmap tensor_map;
    FILE *fp;
    pthread_mutex_t lock;
    bool loaded;
} shard;

typedef struct {
    char *hf_dir;
    weight_map_entry *weights;
    int n_weights;
    hmap weight_map;
    shard *shards;
    int n_shards;
    int cap_shards;
    pthread_mutex_t lock;
} st_db;

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint8_t *data;
    size_t nbytes;
} st_value;

static void st_value_free(st_value *v) {
    free(v->dtype);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void parse_shape(const json_doc *d, int arr_tok, st_info *info, const char *name) {
    if (d->v[arr_tok].type != JT_ARRAY) {
        fprintf(stderr, "error: bad shape for %s\n", name);
        exit(1);
    }
    int nd = 0;
    for (int i = arr_tok + 1; i < d->len && d->v[i].parent == arr_tok; i = json_skip(d, i)) {
        if (nd >= MAX_DIMS) die("too many safetensors dimensions");
        info->shape[nd++] = json_i64(d, i);
    }
    info->n_dims = nd;
}

static int db_find_shard(st_db *db, const char *file) {
    for (int i = 0; i < db->n_shards; i++) {
        if (strcmp(db->shards[i].file, file) == 0) return i;
    }
    if (db->n_shards == db->cap_shards) {
        db->cap_shards = db->cap_shards ? db->cap_shards * 2 : 32;
        db->shards = xrealloc(db->shards, (size_t)db->cap_shards * sizeof(db->shards[0]));
    }
    shard *s = &db->shards[db->n_shards];
    memset(s, 0, sizeof(*s));
    s->file = xstrdup(file);
    s->path = path_join(db->hf_dir, file);
    pthread_mutex_init(&s->lock, NULL);
    return db->n_shards++;
}

static void shard_add_tensor(shard *s, char *name, st_info info) {
    if (s->n_tensors == s->cap_tensors) {
        s->cap_tensors = s->cap_tensors ? s->cap_tensors * 2 : 256;
        s->tensors = xrealloc(s->tensors, (size_t)s->cap_tensors * sizeof(s->tensors[0]));
    }
    s->tensors[s->n_tensors++] = (tensor_entry){ .name = name, .info = info };
}

static void shard_load(shard *s) {
    if (s->loaded) return;
    FILE *fp = fopen(s->path, "rb");
    if (!fp) die_errno("open", s->path);
    uint64_t header_len = read_u64_le_fp(fp, "safetensors header length");
    char *header = xmalloc((size_t)header_len + 1);
    if (fread(header, 1, (size_t)header_len, fp) != (size_t)header_len) die_errno("read header", s->path);
    header[header_len] = '\0';
    s->data_base = 8 + header_len;

    json_doc d = json_parse_text(header, (size_t)header_len);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("bad safetensors header");
    for (int i = 1; i < d.len && d.v[i].parent == 0;) {
        int k = i;
        int v = i + 1;
        if (v >= d.len || d.v[v].parent != 0) die("bad safetensors header object");
        if (!json_tok_eq(&d, k, "__metadata__")) {
            char *name = json_strdup_tok(&d, k);
            st_info info = {0};
            int dtype = json_obj_get(&d, v, "dtype");
            int shape = json_obj_get(&d, v, "shape");
            int offsets = json_obj_get(&d, v, "data_offsets");
            if (dtype < 0 || shape < 0 || offsets < 0) die("bad safetensors tensor entry");
            info.dtype = json_strdup_tok(&d, dtype);
            parse_shape(&d, shape, &info, name);
            int n_off = 0;
            for (int j = offsets + 1; j < d.len && d.v[j].parent == offsets; j = json_skip(&d, j)) {
                int64_t x = json_i64(&d, j);
                if (n_off == 0) info.begin = (uint64_t)x;
                else if (n_off == 1) info.end = (uint64_t)x;
                n_off++;
            }
            if (n_off != 2) die("bad safetensors data_offsets");
            shard_add_tensor(s, name, info);
        }
        i = json_skip(&d, v);
    }
    char **keys = xmalloc((size_t)s->n_tensors * sizeof(keys[0]));
    for (int i = 0; i < s->n_tensors; i++) keys[i] = s->tensors[i].name;
    hmap_build(&s->tensor_map, keys, s->n_tensors);
    free(keys);
    json_free(&d);
    free(header);
    s->fp = fp;
    s->loaded = true;
}

static void db_open(st_db *db, const char *hf_dir) {
    memset(db, 0, sizeof(*db));
    pthread_mutex_init(&db->lock, NULL);
    db->hf_dir = xstrdup(hf_dir);
    char *index_path = path_join(hf_dir, "model.safetensors.index.json");
    size_t len = 0;
    char *text = read_file(index_path, &len);
    json_doc d = json_parse_text(text, len);
    int weight_map = json_obj_get(&d, 0, "weight_map");
    if (weight_map < 0 || d.v[weight_map].type != JT_OBJECT) die("safetensors index has no weight_map");

    int cap = 4096;
    db->weights = xmalloc((size_t)cap * sizeof(db->weights[0]));
    for (int i = weight_map + 1; i < d.len && d.v[i].parent == weight_map;) {
        int k = i;
        int v = i + 1;
        if (db->n_weights == cap) {
            cap *= 2;
            db->weights = xrealloc(db->weights, (size_t)cap * sizeof(db->weights[0]));
        }
        db->weights[db->n_weights].name = json_strdup_tok(&d, k);
        db->weights[db->n_weights].file = json_strdup_tok(&d, v);
        db->n_weights++;
        i = json_skip(&d, v);
    }
    char **keys = xmalloc((size_t)db->n_weights * sizeof(keys[0]));
    for (int i = 0; i < db->n_weights; i++) {
        keys[i] = db->weights[i].name;
        db_find_shard(db, db->weights[i].file);
    }
    hmap_build(&db->weight_map, keys, db->n_weights);
    free(keys);
    json_free(&d);
    free(text);
    free(index_path);
}

static void db_close(st_db *db) {
    for (int i = 0; i < db->n_weights; i++) {
        free(db->weights[i].name);
        free(db->weights[i].file);
    }
    for (int i = 0; i < db->n_shards; i++) {
        shard *s = &db->shards[i];
        if (s->fp) fclose(s->fp);
        for (int j = 0; j < s->n_tensors; j++) {
            free(s->tensors[j].name);
            free(s->tensors[j].info.dtype);
        }
        free(s->tensors);
        hmap_free(&s->tensor_map);
        pthread_mutex_destroy(&s->lock);
        free(s->file);
        free(s->path);
    }
    hmap_free(&db->weight_map);
    pthread_mutex_destroy(&db->lock);
    free(db->weights);
    free(db->shards);
    free(db->hf_dir);
    memset(db, 0, sizeof(*db));
}

static bool db_has(const st_db *db, const char *name) {
    return hmap_get(&db->weight_map, name) >= 0;
}

static tensor_entry *db_tensor(st_db *db, const char *name, shard **shard_out) {
    pthread_mutex_lock(&db->lock);
    int wi = hmap_get(&db->weight_map, name);
    if (wi < 0) {
        fprintf(stderr, "error: HF tensor not found: %s\n", name);
        exit(1);
    }
    const char *file = db->weights[wi].file;
    int si = db_find_shard(db, file);
    shard *s = &db->shards[si];
    shard_load(s);
    int ti = hmap_get(&s->tensor_map, name);
    if (ti < 0) {
        fprintf(stderr, "error: HF tensor %s missing from shard %s\n", name, file);
        exit(1);
    }
    if (shard_out) *shard_out = s;
    tensor_entry *te = &s->tensors[ti];
    pthread_mutex_unlock(&db->lock);
    return te;
}

static const st_info *db_info(st_db *db, const char *name) {
    return &db_tensor(db, name, NULL)->info;
}

static st_value db_read(st_db *db, const char *name) {
    shard *s = NULL;
    tensor_entry *te = db_tensor(db, name, &s);
    const size_t nbytes = (size_t)(te->info.end - te->info.begin);
    st_value v = {0};
    v.dtype = xstrdup(te->info.dtype);
    v.n_dims = te->info.n_dims;
    memcpy(v.shape, te->info.shape, sizeof(v.shape));
    v.nbytes = nbytes;
    v.data = xmalloc(nbytes);
    pthread_mutex_lock(&s->lock);
    if (fseeko(s->fp, (off_t)(s->data_base + te->info.begin), SEEK_SET) != 0) die_errno("seek", s->path);
    if (nbytes && fread(v.data, 1, nbytes, s->fp) != nbytes) die_errno("read tensor", s->path);
    pthread_mutex_unlock(&s->lock);
    return v;
}

/* =====
 * DeepSeek V4 data conversion
 */

static float e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float e4m3fn_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7f) return 0.0f;
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float value = exp == 0 ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

static float bf16_to_f32_bits(uint16_t bits) {
    return ds4q_bf16_to_f32(bits);
}

static int64_t value_nelements(const st_value *v) {
    int64_t n = 1;
    for (int i = 0; i < v->n_dims; i++) n *= v->shape[i];
    return n;
}

static float *tensor_to_f32(const st_value *t, int64_t *n_out) {
    const int64_t n = value_nelements(t);
    float *out = xmalloc((size_t)n * sizeof(float));
    if (strcmp(t->dtype, "F32") == 0) {
        if (t->nbytes != (size_t)n * sizeof(float)) die("bad F32 byte size");
        memcpy(out, t->data, t->nbytes);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad BF16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32_bits(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad F16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = ds4q_f16_to_f32(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F8_E4M3") == 0) {
        if (t->nbytes != (size_t)n) die("bad F8_E4M3 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = e4m3fn_to_f32(t->data[i]);
    } else {
        fprintf(stderr, "error: cannot convert HF dtype directly: %s\n", t->dtype);
        exit(1);
    }
    if (n_out) *n_out = n;
    return out;
}

static float *dequant_fp8_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    if (strcmp(w->dtype, "F8_E4M3") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP8 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP8 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t in_dim = w->shape[1];
    const int64_t block_out = 128;
    const int64_t block_in = 128;
    if (out_dim % block_out || in_dim % block_in) die("FP8 dims are not divisible by 128");
    const int64_t scale_rows = out_dim / block_out;
    const int64_t scale_cols = in_dim / block_in;
    if (scale->shape[0] != scale_rows || scale->shape[1] != scale_cols) die("FP8 scale shape mismatch");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t ob = 0; ob < scale_rows; ob++) {
        for (int64_t ib = 0; ib < scale_cols; ib++) {
            const float s = e8m0_to_f32(scale->data[(size_t)ob * (size_t)scale_cols + (size_t)ib]);
            for (int64_t r = 0; r < block_out; r++) {
                const int64_t row = ob * block_out + r;
                const size_t base = (size_t)row * (size_t)in_dim + (size_t)ib * (size_t)block_in;
                for (int64_t c = 0; c < block_in; c++) {
                    out[base + (size_t)c] = e4m3fn_to_f32(w->data[base + (size_t)c]) * s;
                }
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

static float *dequant_fp4_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    static const float fp4_table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    if (strcmp(w->dtype, "I8") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP4 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP4 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t packed_in = w->shape[1];
    const int64_t in_dim = packed_in * 2;
    if (in_dim % 32) die("FP4 in_dim is not divisible by 32");
    const int64_t n_blocks = in_dim / 32;
    if (scale->shape[0] != out_dim || scale->shape[1] != n_blocks) die("FP4 scale shape mismatch");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t b = 0; b < n_blocks; b++) {
            const float s = e8m0_to_f32(scale->data[(size_t)r * (size_t)n_blocks + (size_t)b]);
            const size_t wbase = ((size_t)r * (size_t)n_blocks + (size_t)b) * 16;
            const size_t obase = (size_t)r * (size_t)in_dim + (size_t)b * 32;
            for (int64_t j = 0; j < 16; j++) {
                const uint8_t q = w->data[wbase + (size_t)j];
                out[obase + (size_t)(2*j + 0)] = fp4_table[q & 0x0f] * s;
                out[obase + (size_t)(2*j + 1)] = fp4_table[(q >> 4) & 0x0f] * s;
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

/* =====
 * Imatrix
 */

typedef struct {
    char *name;
    float *values;
    int n_values;
} imatrix_entry;

typedef struct {
    char *file;
    char *dataset;
    imatrix_entry *entries;
    int n_entries;
    hmap map;
    int chunks;
    bool strict;
} imatrix_store;

static void imatrix_load(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    im->strict = strict;
    im->chunks = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    int32_t n_entries = read_i32_fp(fp, "imatrix entry count");
    if (n_entries < 1) die("imatrix has no entries");
    im->entries = xcalloc((size_t)n_entries, sizeof(im->entries[0]));
    im->n_entries = n_entries;
    for (int i = 0; i < n_entries; i++) {
        int32_t len = read_i32_fp(fp, "imatrix name length");
        if (len <= 0 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = '\0';
        int32_t ncall = read_i32_fp(fp, "imatrix calls");
        int32_t nval = read_i32_fp(fp, "imatrix values");
        if (nval < 1) die("bad imatrix value count");
        float *values = xmalloc((size_t)nval * sizeof(float));
        if (fread(values, sizeof(float), (size_t)nval, fp) != (size_t)nval) die("short imatrix value read");
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) values[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(values[j])) die("non-finite imatrix value");
        }
        im->entries[i] = (imatrix_entry){ .name = name, .values = values, .n_values = nval };
    }
    if (fgetc(fp) != EOF) {
        if (fseeko(fp, -1, SEEK_CUR) == 0) {
            im->chunks = read_i32_fp(fp, "imatrix chunks");
            int32_t dlen = read_i32_fp(fp, "imatrix dataset length");
            if (dlen > 0 && dlen < (1 << 20)) {
                im->dataset = xmalloc((size_t)dlen + 1);
                if (fread(im->dataset, 1, (size_t)dlen, fp) == (size_t)dlen) {
                    im->dataset[dlen] = '\0';
                } else {
                    free(im->dataset);
                    im->dataset = NULL;
                }
            }
        }
    }
    fclose(fp);
    char **keys = xmalloc((size_t)n_entries * sizeof(keys[0]));
    for (int i = 0; i < n_entries; i++) keys[i] = im->entries[i].name;
    hmap_build(&im->map, keys, n_entries);
    free(keys);
    fprintf(stderr, "loaded imatrix %s: %d entries%s%s\n",
            path, n_entries, im->dataset ? ", dataset=" : "", im->dataset ? im->dataset : "");
}

static bool imatrix_enabled(const imatrix_store *im) {
    return im && im->n_entries > 0;
}

static const float *imatrix_find(
        const imatrix_store *im,
        const char **names,
        int n_names,
        int64_t ncols,
        int expert_id,
        int n_experts) {
    if (!imatrix_enabled(im)) return NULL;
    char tmp[4096];
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < n_names; i++) {
            if (!names[i]) continue;
            const char *candidate = names[i];
            if (expert_id >= 0 && pass < 2) {
                snprintf(tmp, sizeof(tmp), "%s.expert%s%d", names[i], pass == 0 ? "." : "_", expert_id);
                candidate = tmp;
            } else if (pass < 2) {
                continue;
            }
            int idx = hmap_get(&im->map, candidate);
            if (idx < 0) continue;
            const imatrix_entry *e = &im->entries[idx];
            if ((int64_t)e->n_values == ncols) return e->values;
            if (expert_id >= 0 && n_experts > 0 && (int64_t)e->n_values == ncols * (int64_t)n_experts) {
                return e->values + (size_t)expert_id * (size_t)ncols;
            }
            fprintf(stderr, "error: imatrix size mismatch for %s: got %d expected %" PRId64 "\n",
                    candidate, e->n_values, ncols);
            exit(1);
        }
    }
    if (im->strict) {
        fprintf(stderr, "error: missing imatrix entry for %s\n", names[0] ? names[0] : "(unnamed)");
        exit(1);
    }
    return NULL;
}

static void imatrix_free(imatrix_store *im) {
    for (int i = 0; i < im->n_entries; i++) {
        free(im->entries[i].name);
        free(im->entries[i].values);
    }
    free(im->entries);
    free(im->file);
    free(im->dataset);
    hmap_free(&im->map);
    memset(im, 0, sizeof(*im));
}

/* =====
 * GGUF tensor mapping and quantization policy
 */

typedef enum { EXP_NONE, EXP_W1, EXP_W2, EXP_W3 } expert_part;

typedef struct {
    bool is_expert;
    int layer;
    expert_part part;
} expert_tensor;

static expert_tensor parse_expert_tensor(const char *name) {
    expert_tensor e = {0};
    int layer = -1;
    char kind[16];
    int rest = 0;
    if ((sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2 ||
         sscanf(name, "mtp.%d.ffn_%15[^_]_exps.weight%n", &layer, kind, &rest) == 2)
        && rest == (int)strlen(name))
    {
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            e.is_expert = true;
            e.layer = layer;
            e.part = strcmp(kind, "gate") == 0 ? EXP_W1 : strcmp(kind, "down") == 0 ? EXP_W2 : EXP_W3;
        }
    }
    return e;
}

static const char *expert_part_name(expert_part p) {
    switch (p) {
        case EXP_W1: return "w1";
        case EXP_W2: return "w2";
        case EXP_W3: return "w3";
        default: die("bad expert part");
    }
    return "";
}

typedef struct {
    const char *gguf;
    const char *hf;
} name_map;

static const name_map top_map[] = {
    { "token_embd.weight",      "embed.weight" },
    { "output_norm.weight",     "norm.weight" },
    { "output.weight",          "head.weight" },
    { "output_hc_base.weight",  "hc_head_base" },
    { "output_hc_fn.weight",    "hc_head_fn" },
    { "output_hc_scale.weight", "hc_head_scale" },
};

static const name_map layer_map[] = {
    { "hc_attn_base.weight",              "hc_attn_base" },
    { "hc_attn_fn.weight",                "hc_attn_fn" },
    { "hc_attn_scale.weight",             "hc_attn_scale" },
    { "hc_ffn_base.weight",               "hc_ffn_base" },
    { "hc_ffn_fn.weight",                 "hc_ffn_fn" },
    { "hc_ffn_scale.weight",              "hc_ffn_scale" },
    { "attn_sinks.weight",                "attn.attn_sink" },
    { "attn_q_a.weight",                  "attn.wq_a.weight" },
    { "attn_q_b.weight",                  "attn.wq_b.weight" },
    { "attn_q_a_norm.weight",             "attn.q_norm.weight" },
    { "attn_kv.weight",                   "attn.wkv.weight" },
    { "attn_kv_a_norm.weight",            "attn.kv_norm.weight" },
    { "attn_output_a.weight",             "attn.wo_a.weight" },
    { "attn_output_b.weight",             "attn.wo_b.weight" },
    { "attn_compressor_ape.weight",       "attn.compressor.ape" },
    { "attn_compressor_kv.weight",        "attn.compressor.wkv.weight" },
    { "attn_compressor_gate.weight",      "attn.compressor.wgate.weight" },
    { "attn_compressor_norm.weight",      "attn.compressor.norm.weight" },
    { "indexer.attn_q_b.weight",          "attn.indexer.wq_b.weight" },
    { "indexer.proj.weight",              "attn.indexer.weights_proj.weight" },
    { "indexer_compressor_ape.weight",    "attn.indexer.compressor.ape" },
    { "indexer_compressor_kv.weight",     "attn.indexer.compressor.wkv.weight" },
    { "indexer_compressor_gate.weight",   "attn.indexer.compressor.wgate.weight" },
    { "indexer_compressor_norm.weight",   "attn.indexer.compressor.norm.weight" },
    { "attn_norm.weight",                 "attn_norm.weight" },
    { "ffn_norm.weight",                  "ffn_norm.weight" },
    { "ffn_gate_shexp.weight",            "ffn.shared_experts.w1.weight" },
    { "ffn_up_shexp.weight",              "ffn.shared_experts.w3.weight" },
    { "ffn_down_shexp.weight",            "ffn.shared_experts.w2.weight" },
    { "ffn_gate_inp.weight",              "ffn.gate.weight" },
    { "exp_probs_b.bias",                 "ffn.gate.bias" },
    { "ffn_gate_tid2eid.weight",          "ffn.gate.tid2eid" },
};

static bool g_dspark_source_mode = false;

typedef struct gguf_file gguf_file;
typedef struct {
    uint8_t *data;
    size_t size;
} byte_buf;
static const gguf_file *g_template_file;
static const char *g_template_path;
static byte_buf read_gguf_tensor_data(const gguf_file *g, const char *path, const char *name);

static char *hf_name_for_regular(const char *gguf_name) {
    if (g_dspark_source_mode) {
        if (strcmp(gguf_name, "mtp.2.hc_head_base.weight") == 0) return xstrdup("mtp.2.hc_head_base");
        if (strcmp(gguf_name, "mtp.2.hc_head_fn.weight") == 0) return xstrdup("mtp.2.hc_head_fn");
        if (strcmp(gguf_name, "mtp.2.hc_head_scale.weight") == 0) return xstrdup("mtp.2.hc_head_scale");
        if (strcmp(gguf_name, "mtp.2.norm.weight") == 0 ||
            strcmp(gguf_name, "mtp.0.main_proj.weight") == 0 ||
            strcmp(gguf_name, "mtp.0.main_norm.weight") == 0 ||
            strcmp(gguf_name, "mtp.2.markov_head.markov_w1.weight") == 0 ||
            strcmp(gguf_name, "mtp.2.markov_head.markov_w2.weight") == 0 ||
            strcmp(gguf_name, "mtp.2.confidence_head.proj.weight") == 0) {
            return xstrdup(gguf_name);
        }
    }
    for (size_t i = 0; i < sizeof(top_map) / sizeof(top_map[0]); i++) {
        if (strcmp(gguf_name, top_map[i].gguf) == 0) return xstrdup(top_map[i].hf);
    }
    int layer = -1;
    char prefix[16];
    const char *p = gguf_name;
    if (sscanf(p, "%15[^.].%d.", prefix, &layer) != 2) {
        fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
        exit(1);
    }
    const char *rest = strchr(p + 4, '.');
    if (!rest) die("bad layer tensor name");
    rest++;
    const char *layer_prefix = str_starts(gguf_name, "mtp.") ? "mtp" : "layers";
    for (size_t i = 0; i < sizeof(layer_map) / sizeof(layer_map[0]); i++) {
        if (strcmp(rest, layer_map[i].gguf) == 0) {
            char buf[512];
            snprintf(buf, sizeof(buf), "%s.%d.%s", layer_prefix, layer, layer_map[i].hf);
            return xstrdup(buf);
        }
    }
    fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
    exit(1);
}

typedef struct {
    char *prefix;
    ds4q_type type;
} type_override;

typedef struct {
    ds4q_type routed_w1, routed_w2, routed_w3;
    ds4q_type attention_proj, attention, shared, embedding, output, dense;
    type_override *overrides;
    int n_overrides;
} quant_policy;

static bool is_attention_projection(const char *name) {
    return strstr(name, ".attn_kv.weight") || strstr(name, ".attn_q_a.weight") ||
           strstr(name, ".attn_q_b.weight") || strstr(name, ".attn_output_a.weight") ||
           strstr(name, ".attn_output_b.weight");
}

static bool is_attention_tensor(const char *name) {
    return strstr(name, ".attn") || strstr(name, "attn_") || strstr(name, ".indexer") || strstr(name, "indexer_");
}

static bool is_shared_expert(const char *name) {
    return strstr(name, "_shexp.") != NULL;
}

static bool is_output_tensor(const char *name) {
    return str_starts(name, "output.");
}

static bool is_dspark_support_tensor(const char *name) {
    return strcmp(name, "mtp.0.main_proj.weight") == 0 ||
           strcmp(name, "mtp.0.main_norm.weight") == 0 ||
           strcmp(name, "mtp.2.markov_head.markov_w1.weight") == 0 ||
           strcmp(name, "mtp.2.markov_head.markov_w2.weight") == 0 ||
           strcmp(name, "mtp.2.confidence_head.proj.weight") == 0;
}

static char *dspark_output_tensor_name(const char *name) {
    return xstrdup(name);
}

static char *dspark_template_tensor_name(const char *name) {
    if (!g_dspark_source_mode || !str_starts(name, "mtp.")) return xstrdup(name);
    if (strcmp(name, "mtp.2.norm.weight") == 0 ||
        strcmp(name, "mtp.2.hc_head_base.weight") == 0 ||
        strcmp(name, "mtp.2.hc_head_fn.weight") == 0 ||
        strcmp(name, "mtp.2.hc_head_scale.weight") == 0 ||
        strcmp(name, "mtp.0.main_proj.weight") == 0 ||
        strcmp(name, "mtp.0.main_norm.weight") == 0 ||
        strcmp(name, "mtp.2.markov_head.markov_w1.weight") == 0 ||
        strcmp(name, "mtp.2.markov_head.markov_w2.weight") == 0 ||
        strcmp(name, "mtp.2.confidence_head.proj.weight") == 0) {
        return xstrdup(name);
    }
    char *mapped = xstrdup(name);
    memcpy(mapped, "blk.", 4);
    return mapped;
}

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[DS4Q_MAX_DIMS];
    ds4q_type type;
    uint64_t old_offset;
    uint64_t new_offset;
    size_t size;
} tensor_meta;

static int tensor_n_dims(const tensor_meta *t) {
    int n = t->n_dims;
    while (n > 1 && t->ne[n - 1] == 1) n--;
    return n;
}

static ds4q_type policy_type(const quant_policy *p, const char *name, const tensor_meta *tmpl) {
    if (strcmp(name, "output_hc_base.weight") == 0) return DS4Q_TYPE_F32;
    if (strcmp(name, "output_hc_fn.weight") == 0) return DS4Q_TYPE_F16;
    if (strcmp(name, "output_hc_scale.weight") == 0) return DS4Q_TYPE_F32;
    if (strcmp(name, "mtp.2.norm.weight") == 0) return DS4Q_TYPE_F32;
    if (strcmp(name, "mtp.2.hc_head_base.weight") == 0) return DS4Q_TYPE_F32;
    if (strcmp(name, "mtp.2.hc_head_fn.weight") == 0) return DS4Q_TYPE_F16;
    if (strcmp(name, "mtp.2.hc_head_scale.weight") == 0) return DS4Q_TYPE_F32;
    if (strcmp(name, "mtp.0.main_proj.weight") == 0) return DS4Q_TYPE_F16;
    if (strcmp(name, "mtp.0.main_norm.weight") == 0) return DS4Q_TYPE_F32;
    if (strcmp(name, "mtp.2.markov_head.markov_w1.weight") == 0) return DS4Q_TYPE_BF16;
    if (strcmp(name, "mtp.2.markov_head.markov_w2.weight") == 0) return DS4Q_TYPE_BF16;
    if (strcmp(name, "mtp.2.confidence_head.proj.weight") == 0) return DS4Q_TYPE_BF16;
    for (int i = 0; i < p->n_overrides; i++) {
        if (strcmp(name, p->overrides[i].prefix) == 0 || str_starts(name, p->overrides[i].prefix)) {
            return p->overrides[i].type;
        }
    }
    expert_tensor e = parse_expert_tensor(name);
    if (e.is_expert) {
        if (e.part == EXP_W1 && p->routed_w1 != DS4Q_TYPE_COUNT) return p->routed_w1;
        if (e.part == EXP_W2 && p->routed_w2 != DS4Q_TYPE_COUNT) return p->routed_w2;
        if (e.part == EXP_W3 && p->routed_w3 != DS4Q_TYPE_COUNT) return p->routed_w3;
        return tmpl->type;
    }
    if (tmpl->type != DS4Q_TYPE_F32 && tmpl->type != DS4Q_TYPE_F16 &&
        tmpl->type != DS4Q_TYPE_BF16 && !ds4q_can_quantize(tmpl->type)) {
        return tmpl->type;
    }
    if (tensor_n_dims(tmpl) <= 1) return tmpl->type;
    if (strcmp(name, "token_embd.weight") == 0 && p->embedding != DS4Q_TYPE_COUNT) return p->embedding;
    if (is_output_tensor(name) && p->output != DS4Q_TYPE_COUNT) return p->output;
    if (is_shared_expert(name) && p->shared != DS4Q_TYPE_COUNT) return p->shared;
    if (is_attention_projection(name) && p->attention_proj != DS4Q_TYPE_COUNT) return p->attention_proj;
    if (is_attention_tensor(name) && p->attention != DS4Q_TYPE_COUNT) return p->attention;
    if (p->dense != DS4Q_TYPE_COUNT) return p->dense;
    return tmpl->type;
}

static ds4q_type parse_type(const char *raw) {
    char wanted[64];
    size_t n = 0;
    for (const char *p = raw; *p && n + 1 < sizeof(wanted); p++) {
        if (*p != '-' && *p != '_') wanted[n++] = (char)tolower((unsigned char)*p);
    }
    wanted[n] = '\0';
    if (strcmp(wanted, "copy") == 0 || strcmp(wanted, "template") == 0) return DS4Q_TYPE_COUNT;
    for (int i = 0; i < DS4Q_TYPE_COUNT; i++) {
        char name[64];
        size_t m = 0;
        const char *tn = ds4q_type_name((ds4q_type)i);
        if (!tn) continue;
        for (const char *p = tn; *p && m + 1 < sizeof(name); p++) {
            if (*p != '-' && *p != '_') name[m++] = (char)tolower((unsigned char)*p);
        }
        name[m] = '\0';
        if (strcmp(name, wanted) == 0) return (ds4q_type)i;
    }
    fprintf(stderr, "error: unknown quant type: %s\n", raw);
    exit(1);
}

static bool is_quantizable_target(ds4q_type type) {
    return type == DS4Q_TYPE_F32 || type == DS4Q_TYPE_F16 || type == DS4Q_TYPE_BF16 || ds4q_can_quantize(type);
}

/* =====
 * Tensor generation
 */

static byte_buf f32_to_type(const float *src, int64_t n, ds4q_type type, int64_t ncols, const float *imat) {
    if (ncols <= 0 || n % ncols != 0) die("bad ncols for tensor conversion");
    byte_buf out = {0};
    if (type == DS4Q_TYPE_F32) {
        out.size = (size_t)n * sizeof(float);
        out.data = xmalloc(out.size);
        memcpy(out.data, src, out.size);
        return out;
    }
    if (type == DS4Q_TYPE_F16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_f16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (type == DS4Q_TYPE_BF16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_bf16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (!ds4q_can_quantize(type)) die("unsupported quant target type");
    if (ncols % ds4q_block_size(type) != 0) die("ncols is not divisible by quant block size");
    const int64_t nrows = n / ncols;
    out.size = (size_t)nrows * ds4q_row_size(type, ncols);
    out.data = xmalloc(out.size);

    float *synthetic = NULL;
    const float *im_ptr = imat;
    if (!im_ptr && ds4q_requires_imatrix(type)) {
        synthetic = xcalloc((size_t)ncols, sizeof(float));
        for (int64_t r = 0; r < nrows; r++) {
            const float *row = src + (size_t)r * (size_t)ncols;
            for (int64_t c = 0; c < ncols; c++) synthetic[c] += row[c] * row[c];
        }
        im_ptr = synthetic;
    }
    size_t written = ds4q_quantize_chunk(type, src, out.data, 0, nrows, ncols, im_ptr);
    free(synthetic);
    if (written != out.size) die("ds4q_quantize_chunk wrote unexpected byte count");
    return out;
}

static byte_buf i64_to_i32(const st_value *src) {
    if (strcmp(src->dtype, "I64") != 0) die("expected I64 source for I32 tensor");
    const int64_t n = value_nelements(src);
    if (src->nbytes != (size_t)n * sizeof(int64_t)) die("bad I64 byte size");
    byte_buf out = { .size = (size_t)n * sizeof(int32_t), .data = xmalloc((size_t)n * sizeof(int32_t)) };
    int32_t *dst = (int32_t *)out.data;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = load_i64_le(src->data + (size_t)i * 8);
        if (v < INT32_MIN || v > INT32_MAX) die("I64 value out of I32 range");
        dst[i] = (int32_t)v;
    }
    return out;
}

static size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims) {
    size_t nbytes = ds4q_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) nbytes *= (size_t)ne[i];
    return nbytes;
}

static void check_reversed_shape(const char *gguf_name, const st_info *info, const tensor_meta *tmpl) {
    int nd = tensor_n_dims(tmpl);
    if (info->n_dims != nd) {
        fprintf(stderr, "error: rank mismatch for %s\n", gguf_name);
        exit(1);
    }
    for (int i = 0; i < nd; i++) {
        if (tmpl->ne[i] != info->shape[nd - 1 - i]) {
            fprintf(stderr, "error: shape mismatch for %s\n", gguf_name);
            exit(1);
        }
    }
}

static void check_exact_shape(const char *gguf_name, const st_info *info, const tensor_meta *tmpl) {
    int nd = tensor_n_dims(tmpl);
    if (info->n_dims != nd) {
        fprintf(stderr, "error: rank mismatch for %s\n", gguf_name);
        exit(1);
    }
    for (int i = 0; i < nd; i++) {
        if (tmpl->ne[i] != info->shape[i]) {
            fprintf(stderr, "error: shape mismatch for %s\n", gguf_name);
            exit(1);
        }
    }
}

static byte_buf generate_regular(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                 ds4q_type target, const imatrix_store *imatrix) {
    if (g_template_file && target == tmpl->type) {
        return read_gguf_tensor_data(g_template_file, g_template_path, gguf_name);
    }
    /*
     * DSpark-style checkpoints already use canonical HF tensor names for the
     * support draft namespace. Try the exact name first so a DSpark template
     * can mirror the HF state dict directly; otherwise fall back to the legacy
     * Flash/PRO name translation.
     */
    char *hf_name = NULL;
    if (db_has(db, gguf_name)) {
        hf_name = xstrdup(gguf_name);
    } else {
        hf_name = hf_name_for_regular(gguf_name);
    }
    if (!db_has(db, hf_name)) {
        if (g_dspark_source_mode && g_template_file && target == tmpl->type) {
            char *tmpl_name = dspark_template_tensor_name(gguf_name);
            byte_buf b = read_gguf_tensor_data(g_template_file, g_template_path, tmpl_name);
            free(tmpl_name);
            free(hf_name);
            return b;
        }
    }
    tensor_entry *te = db_tensor(db, hf_name, NULL);
    if (g_dspark_source_mode && strcmp(gguf_name, "mtp.2.confidence_head.proj.weight") == 0) {
        if (!(te->info.n_dims == 1 && tensor_n_dims(tmpl) == 1 && te->info.shape[0] == tmpl->ne[0]) &&
            !(te->info.n_dims == 2 && tensor_n_dims(tmpl) == 1 && te->info.shape[0] == 1 && te->info.shape[1] == tmpl->ne[0])) {
            fprintf(stderr, "error: shape mismatch for %s\n", gguf_name);
            exit(1);
        }
    } else if (g_dspark_source_mode && is_dspark_support_tensor(gguf_name) && tensor_n_dims(tmpl) == 1) {
        check_exact_shape(gguf_name, &te->info, tmpl);
    } else {
        check_reversed_shape(gguf_name, &te->info, tmpl);
    }
    if (target == DS4Q_TYPE_I32) {
        st_value sv = db_read(db, hf_name);
        byte_buf b = i64_to_i32(&sv);
        st_value_free(&sv);
        free(hf_name);
        return b;
    }
    if (!is_quantizable_target(target)) die("unsupported regular target type");
    int64_t n = 0;
    float *f32 = NULL;
    if (strcmp(te->info.dtype, "F8_E4M3") == 0) {
        if (!str_ends(hf_name, ".weight")) die("FP8 tensor without .weight suffix");
        char *scale_name = xstrdup(hf_name);
        strcpy(scale_name + strlen(scale_name) - strlen(".weight"), ".scale");
        if (!db_has(db, scale_name)) die("missing FP8 scale tensor");
        st_value w = db_read(db, hf_name);
        st_value s = db_read(db, scale_name);
        f32 = dequant_fp8_weight(&w, &s, &n);
        st_value_free(&w);
        st_value_free(&s);
        free(scale_name);
    } else {
        st_value w = db_read(db, hf_name);
        f32 = tensor_to_f32(&w, &n);
        st_value_free(&w);
    }
    const char *names[2] = { gguf_name, hf_name };
    const float *imat = imatrix_find(imatrix, names, 2, tmpl->ne[0], -1, 0);
    byte_buf b = f32_to_type(f32, n, target, tmpl->ne[0], imat);
    free(f32);
    free(hf_name);
    return b;
}

typedef struct {
    st_db *db;
    const char *gguf_name;
    const tensor_meta *tmpl;
    ds4q_type target;
    int n_experts;
    const imatrix_store *imatrix;
    expert_tensor expert;
    const char *wid;
    int64_t ncols;
    int64_t nrows;
    size_t per_expert;
    byte_buf *out;
    int next;
    int done;
    pthread_mutex_t lock;
} expert_job;

static void generate_one_expert(expert_job *j, int xid) {
    char prefix[256];
    const char *layer_prefix = str_starts(j->gguf_name, "mtp.") ? "mtp" : "layers";
    snprintf(prefix, sizeof(prefix), "%s.%d.ffn.experts.%d.%s",
             layer_prefix,
             j->expert.layer, xid, j->wid);
    char weight_name[320];
    char scale_name[320];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", prefix);
    snprintf(scale_name, sizeof(scale_name), "%s.scale", prefix);
    st_value w = db_read(j->db, weight_name);
    st_value s = db_read(j->db, scale_name);
    if (w.n_dims != 2 || w.shape[0] != j->nrows || w.shape[1] * 2 != j->ncols) die("expert shape mismatch");
    int64_t n = 0;
    float *f32 = dequant_fp4_weight(&w, &s, &n);
    const char *names[3] = { j->gguf_name, weight_name, NULL };
    const float *imat = imatrix_find(j->imatrix, names, 2, j->ncols, xid, j->n_experts);
    byte_buf q = f32_to_type(f32, n, j->target, j->ncols, imat);
    if (q.size != j->per_expert) die("expert quantized size mismatch");
    memcpy(j->out->data + (size_t)xid * j->per_expert, q.data, q.size);
    free(q.data);
    free(f32);
    st_value_free(&w);
    st_value_free(&s);
}

static void *expert_worker(void *arg) {
    expert_job *j = arg;
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int xid = j->next++;
        pthread_mutex_unlock(&j->lock);
        if (xid >= j->n_experts) break;
        generate_one_expert(j, xid);
        pthread_mutex_lock(&j->lock);
        int done = ++j->done;
        if (done % 32 == 0 || done == j->n_experts) {
            fprintf(stderr, "generate_expert_tensor: layer %d %s %d/%d experts\n",
                    j->expert.layer, j->wid, done, j->n_experts);
        }
        pthread_mutex_unlock(&j->lock);
    }
    return NULL;
}

static byte_buf generate_expert(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix) {
    expert_tensor e = parse_expert_tensor(gguf_name);
    if (!e.is_expert) die("not an expert tensor");
    if (g_template_file && target == tmpl->type) {
        return read_gguf_tensor_data(g_template_file, g_template_path, gguf_name);
    }
    if (g_dspark_source_mode && g_template_file && target == tmpl->type && str_starts(gguf_name, "mtp.")) {
        char probe[320];
        snprintf(probe, sizeof(probe), "mtp.%d.ffn.experts.0.%s.weight", e.layer, expert_part_name(e.part));
        if (!db_has(db, probe)) {
            return read_gguf_tensor_data(g_template_file, g_template_path, gguf_name);
        }
    }
    if (!is_quantizable_target(target)) die("unsupported expert target type");
    const char *wid = expert_part_name(e.part);
    const int64_t ncols = tmpl->ne[0];
    const int64_t nrows = tmpl->ne[1];
    const size_t per_expert = (size_t)nrows * ds4q_row_size(target, ncols);
    byte_buf out = { .size = per_expert * (size_t)n_experts, .data = xmalloc(per_expert * (size_t)n_experts) };
    ds4q_quantize_init(target);
    int worker_count = n_threads > 0 ? n_threads : 8;
    if (worker_count < 1) worker_count = 1;
    if (worker_count > n_experts) worker_count = n_experts;
    fprintf(stderr, "generate_expert_tensor: layer %d %s using %d worker%s\n",
            e.layer, wid, worker_count, worker_count == 1 ? "" : "s");
    expert_job job = {
        .db = db, .gguf_name = gguf_name, .tmpl = tmpl, .target = target,
        .n_experts = n_experts, .imatrix = imatrix, .expert = e, .wid = wid,
        .ncols = ncols, .nrows = nrows, .per_expert = per_expert, .out = &out,
    };
    pthread_mutex_init(&job.lock, NULL);
    pthread_t *threads = xcalloc((size_t)worker_count, sizeof(threads[0]));
    for (int i = 1; i < worker_count; i++) pthread_create(&threads[i], NULL, expert_worker, &job);
    expert_worker(&job);
    for (int i = 1; i < worker_count; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&job.lock);
    free(threads);
    return out;
}

static byte_buf generate_tensor(st_db *db, const char *name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix) {
    if (parse_expert_tensor(name).is_expert) {
        return generate_expert(db, name, tmpl, target, n_experts, n_threads, imatrix);
    }
    return generate_regular(db, name, tmpl, target, imatrix);
}

/* =====
 * Minimal GGUF reader/writer
 *
 * GGUF metadata is copied as raw KV records from the template.  Tensor infos
 * are rewritten with the new target types and offsets.  This keeps the tool C
 * only and independent from general-purpose GGUF libraries.
 */

typedef struct {
    size_t start;
    size_t end;
} byte_span;

typedef struct gguf_file {
    char *path;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint8_t *kv_raw;
    size_t kv_raw_len;
    size_t alignment;
    int n_experts;
    size_t data_offset;
    tensor_meta *tensors;
    hmap tensor_map;
} gguf_file;

static const gguf_file *g_template_file = NULL;

typedef struct {
    tensor_meta *tensors;
    uint64_t n_tensors;
    uint64_t n_kv_extra;
    bool has_dspark_metadata;
    size_t meta_size;
    size_t data_offset;
    size_t tensor_bytes;
    size_t alignment;
} output_context;

typedef struct {
    bool enabled;
    uint32_t version;
    char algorithm[16];
    char carrier_kind[32];
    char model_id[64];
    char backend_compat[32];
    uint32_t recommended_draft_tokens;
    uint32_t n_mtp_layers;
    uint32_t block_size;
    uint32_t noise_token_id;
    uint32_t markov_rank;
    uint32_t target_layer_ids[3];
    bool abliterated;
} dspark_emit_metadata;

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static char *read_gguf_string_fp(FILE *fp) {
    uint64_t n = read_u64_le_fp(fp, "GGUF string length");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static void skip_bytes_fp(FILE *fp, uint64_t n) {
    if (fseeko(fp, (off_t)n, SEEK_CUR) != 0) die("GGUF seek failed");
}

static void skip_gguf_value_fp(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64_le_fp(fp, "GGUF string length");
        skip_bytes_fp(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32_le_fp(fp, "GGUF array type");
        uint64_t n = read_u64_le_fp(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64_le_fp(fp, "GGUF array string length");
                skip_bytes_fp(fp, len);
            }
        } else {
            size_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type");
            skip_bytes_fp(fp, n * sz);
        }
        return;
    }
    size_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF value type");
    skip_bytes_fp(fp, sz);
}

static size_t gguf_string_size(const char *s) {
    return sizeof(uint64_t) + strlen(s);
}

static void write_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u64 failed");
}

static void write_f32(FILE *fp, float v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write f32 failed");
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static bool is_imatrix_kv_key(const char *key) {
    return str_starts(key, "quantize.imatrix.");
}

static size_t extra_imatrix_kv_size(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    size_t n = 0;
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_FILE) + 4 + gguf_string_size(im->file);
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES) + 4 + 8;
    if (im->dataset) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_DATASET) + 4 + gguf_string_size(im->dataset);
    if (im->chunks > 0) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS) + 4 + 8;
    return n;
}

static uint64_t extra_imatrix_kv_count(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    return 2 + (im->dataset ? 1 : 0) + (im->chunks > 0 ? 1 : 0);
}

static size_t extra_dspark_kv_size(const dspark_emit_metadata *dspark) {
    if (!dspark || !dspark->enabled) return 0;
    size_t n = 0;
    n += gguf_string_size("deepseek4.dspark.algorithm") + 4 + gguf_string_size(dspark->algorithm);
    n += gguf_string_size("deepseek4.dspark.carrier_kind") + 4 + gguf_string_size(dspark->carrier_kind);
    n += gguf_string_size("deepseek4.dspark.model_id") + 4 + gguf_string_size(dspark->model_id);
    n += gguf_string_size("deepseek4.dspark.backend_compat") + 4 + gguf_string_size(dspark->backend_compat);
    n += gguf_string_size("deepseek4.dspark.recommended_draft_tokens") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.version") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.n_mtp_layers") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.block_size") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.noise_token_id") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.markov_rank") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.target_layer_ids.0") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.target_layer_ids.1") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.target_layer_ids.2") + 4 + 4;
    n += gguf_string_size("deepseek4.dspark.abliterated") + 4 + 1;
    return n;
}

static uint64_t extra_dspark_kv_count(const dspark_emit_metadata *dspark) {
    return dspark && dspark->enabled ? 14 : 0;
}

static void write_dspark_kvs(FILE *fp, const dspark_emit_metadata *dspark) {
    if (!dspark || !dspark->enabled) return;

    write_gguf_string(fp, "deepseek4.dspark.algorithm");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, dspark->algorithm);

    write_gguf_string(fp, "deepseek4.dspark.carrier_kind");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, dspark->carrier_kind);

    write_gguf_string(fp, "deepseek4.dspark.model_id");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, dspark->model_id);

    write_gguf_string(fp, "deepseek4.dspark.backend_compat");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, dspark->backend_compat);

    write_gguf_string(fp, "deepseek4.dspark.recommended_draft_tokens");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->recommended_draft_tokens);

    write_gguf_string(fp, "deepseek4.dspark.version");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->version);

    write_gguf_string(fp, "deepseek4.dspark.n_mtp_layers");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->n_mtp_layers);

    write_gguf_string(fp, "deepseek4.dspark.block_size");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->block_size);

    write_gguf_string(fp, "deepseek4.dspark.noise_token_id");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->noise_token_id);

    write_gguf_string(fp, "deepseek4.dspark.markov_rank");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->markov_rank);

    write_gguf_string(fp, "deepseek4.dspark.target_layer_ids.0");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->target_layer_ids[0]);
    write_gguf_string(fp, "deepseek4.dspark.target_layer_ids.1");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->target_layer_ids[1]);
    write_gguf_string(fp, "deepseek4.dspark.target_layer_ids.2");
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, dspark->target_layer_ids[2]);

    write_gguf_string(fp, "deepseek4.dspark.abliterated");
    write_u32(fp, GGUF_TYPE_BOOL);
    uint8_t b = dspark->abliterated ? 1u : 0u;
    if (fwrite(&b, 1, 1, fp) != 1) die("write bool failed");
}

static void dspark_emit_metadata_set_defaults(dspark_emit_metadata *out, const char *carrier_kind) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->enabled = true;
    out->version = 1;
    out->abliterated = true;
    snprintf(out->algorithm, sizeof(out->algorithm), "%s", "dspark");
    snprintf(out->carrier_kind, sizeof(out->carrier_kind), "%s",
             carrier_kind && carrier_kind[0] ? carrier_kind : "official-dspark");
    snprintf(out->model_id, sizeof(out->model_id), "%s", "deepseek-v4-flash-dspark");
    snprintf(out->backend_compat, sizeof(out->backend_compat), "%s", "cuda-spark");
    out->recommended_draft_tokens = 2;
    out->n_mtp_layers = 3;
    out->block_size = 5;
    out->noise_token_id = 128799u;
    out->markov_rank = 256u;
    out->target_layer_ids[0] = 40u;
    out->target_layer_ids[1] = 41u;
    out->target_layer_ids[2] = 42u;
}

static bool json_try_u32_obj(const json_doc *d, int obj, const char *key, uint32_t *out) {
    int tok = json_obj_get(d, obj, key);
    if (tok < 0) return false;
    if (d->v[tok].type != JT_PRIMITIVE) return false;
    int64_t v = json_i64(d, tok);
    if (v < 0 || (uint64_t)v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool json_try_bool_obj(const json_doc *d, int obj, const char *key, bool *out) {
    int tok = json_obj_get(d, obj, key);
    if (tok < 0) return false;
    if (d->v[tok].type != JT_PRIMITIVE) return false;
    const int n = d->v[tok].end - d->v[tok].start;
    const char *s = d->js + d->v[tok].start;
    if (n == 4 && !memcmp(s, "true", 4)) {
        *out = true;
        return true;
    }
    if (n == 5 && !memcmp(s, "false", 5)) {
        *out = false;
        return true;
    }
    if (n == 1 && s[0] >= '0' && s[0] <= '9') {
        *out = s[0] != '0';
        return true;
    }
    if (n == 2 && s[0] == '1' && s[1] == '0') {
        *out = true;
        return true;
    }
    return false;
}

static bool json_try_string_obj(const json_doc *d, int obj, const char *key, char *out, size_t out_cap) {
    int tok = json_obj_get(d, obj, key);
    if (tok < 0) return false;
    if (d->v[tok].type != JT_STRING) return false;
    size_t n = (size_t)(d->v[tok].end - d->v[tok].start);
    if (n >= out_cap) n = out_cap - 1;
    memcpy(out, d->js + d->v[tok].start, n);
    out[n] = '\0';
    return true;
}

static bool load_dspark_emit_metadata(const char *hf_dir, dspark_emit_metadata *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));

    char *config_path = path_join(hf_dir, "config.json");
    FILE *check = fopen(config_path, "rb");
    if (!check) {
        free(config_path);
        return false;
    }
    fclose(check);
    size_t len = 0;
    char *text = read_file(config_path, &len);
    free(config_path);

    json_doc d = json_parse_text(text, len);
    bool ok = d.len > 0 && d.v[0].type == JT_OBJECT;
    int root = 0;
    if (!ok) {
        json_free(&d);
        free(text);
        return false;
    }

    out->enabled = true;
    out->version = 1;
    out->abliterated = true;
    snprintf(out->algorithm, sizeof(out->algorithm), "%s", "dspark");
    snprintf(out->carrier_kind, sizeof(out->carrier_kind), "%s", "official-dspark");
    snprintf(out->model_id, sizeof(out->model_id), "%s", "deepseek-v4-flash-dspark");
    snprintf(out->backend_compat, sizeof(out->backend_compat), "%s", "cuda-spark");
    out->recommended_draft_tokens = 2;
    (void)json_try_string_obj(&d, root, "dspark_algorithm", out->algorithm, sizeof(out->algorithm));
    (void)json_try_string_obj(&d, root, "deepseek4.dspark.algorithm", out->algorithm, sizeof(out->algorithm));
    (void)json_try_string_obj(&d, root, "dspark_carrier_kind", out->carrier_kind, sizeof(out->carrier_kind));
    (void)json_try_string_obj(&d, root, "deepseek4.dspark.carrier_kind", out->carrier_kind, sizeof(out->carrier_kind));
    if (!json_try_u32_obj(&d, root, "dspark_version", &out->version) &&
        !json_try_u32_obj(&d, root, "deepseek4.dspark.version", &out->version)) {
        out->version = 1;
    }
    if (!json_try_string_obj(&d, root, "dspark_model_id", out->model_id, sizeof(out->model_id)) &&
        !json_try_string_obj(&d, root, "deepseek4.dspark.model_id", out->model_id, sizeof(out->model_id))) {
        snprintf(out->model_id, sizeof(out->model_id), "%s", "deepseek-v4-flash-dspark");
    }
    if (!json_try_string_obj(&d, root, "dspark_backend_compat", out->backend_compat, sizeof(out->backend_compat)) &&
        !json_try_string_obj(&d, root, "deepseek4.dspark.backend_compat", out->backend_compat, sizeof(out->backend_compat))) {
        snprintf(out->backend_compat, sizeof(out->backend_compat), "%s", "cuda-spark");
    }
    if (!json_try_u32_obj(&d, root, "dspark_recommended_draft_tokens", &out->recommended_draft_tokens) &&
        !json_try_u32_obj(&d, root, "deepseek4.dspark.recommended_draft_tokens", &out->recommended_draft_tokens)) {
        out->recommended_draft_tokens = 2;
    }
    if (!json_try_u32_obj(&d, root, "n_mtp_layers", &out->n_mtp_layers) &&
        !json_try_u32_obj(&d, root, "deepseek4.dspark.n_mtp_layers", &out->n_mtp_layers)) out->n_mtp_layers = 3;
    if (!json_try_u32_obj(&d, root, "dspark_block_size", &out->block_size) &&
        !json_try_u32_obj(&d, root, "deepseek4.dspark.block_size", &out->block_size)) out->block_size = 5;
    if (!json_try_u32_obj(&d, root, "dspark_noise_token_id", &out->noise_token_id) &&
        !json_try_u32_obj(&d, root, "deepseek4.dspark.noise_token_id", &out->noise_token_id)) out->noise_token_id = 128799u;
    if (!json_try_u32_obj(&d, root, "dspark_markov_rank", &out->markov_rank) &&
        !json_try_u32_obj(&d, root, "deepseek4.dspark.markov_rank", &out->markov_rank)) out->markov_rank = 256u;
    if (!json_try_bool_obj(&d, root, "dspark_abliterated", &out->abliterated) &&
        !json_try_bool_obj(&d, root, "deepseek4.dspark.abliterated", &out->abliterated)) {
        out->abliterated = true;
    }
    if (out->algorithm[0] == '\0') snprintf(out->algorithm, sizeof(out->algorithm), "%s", "dspark");
    if (out->carrier_kind[0] == '\0') snprintf(out->carrier_kind, sizeof(out->carrier_kind), "%s", "official-dspark");
    if (out->recommended_draft_tokens == 0) out->recommended_draft_tokens = 2;
    for (uint32_t i = 0; i < 3; i++) out->target_layer_ids[i] = 40u + i;
    int layers = json_obj_get(&d, root, "dspark_target_layer_ids");
    if (layers < 0) layers = json_obj_get(&d, root, "deepseek4.dspark.target_layer_ids");
    if (layers >= 0 && d.v[layers].type == JT_ARRAY) {
        uint32_t idx = 0;
        for (int i = layers + 1; i < d.len && d.v[i].parent == layers && idx < 3; i = json_skip(&d, i)) {
            int64_t v = json_i64(&d, i);
            if (v >= 0 && (uint64_t)v <= UINT32_MAX) out->target_layer_ids[idx++] = (uint32_t)v;
        }
    }
    json_free(&d);
    free(text);
    return true;
}

static const char *guess_dspark_carrier_kind_from_gguf(const gguf_file *src) {
    if (!src) return "unknown";
    if (hmap_get(&src->tensor_map, "mtp.0.main_proj.weight") >= 0) {
        if (hmap_get(&src->tensor_map, "mtp.2.markov_head.markov_w1.weight") >= 0) {
            return "official-dspark";
        }
        return "dspark-nonseq";
    }
    if (hmap_get(&src->tensor_map, "mtp.0.e_proj.weight") >= 0) {
        return "legacy-mtp";
    }
    return "unknown";
}

static void write_kv_string(FILE *fp, const char *key, const char *value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, value);
}

static void write_kv_u32(FILE *fp, const char *key, uint32_t value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, value);
}

static void write_kv_u64(FILE *fp, const char *key, uint64_t value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_UINT64);
    write_u64(fp, value);
}

static void write_kv_f32(FILE *fp, const char *key, float value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_FLOAT32);
    write_f32(fp, value);
}

static void write_kv_bool(FILE *fp, const char *key, bool value) {
    uint8_t b = value ? 1u : 0u;
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_BOOL);
    if (fwrite(&b, 1, 1, fp) != 1) die("write bool failed");
}

static void write_kv_u32_array(FILE *fp, const char *key, const uint32_t *values, uint64_t n) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_ARRAY);
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u64(fp, n);
    for (uint64_t i = 0; i < n; i++) write_u32(fp, values[i]);
}

static void write_kv_f32_array(FILE *fp, const char *key, const float *values, uint64_t n) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_ARRAY);
    write_u32(fp, GGUF_TYPE_FLOAT32);
    write_u64(fp, n);
    for (uint64_t i = 0; i < n; i++) write_f32(fp, values[i]);
}

static uint32_t config_u32(const json_doc *d, int root, const char *key, uint32_t def) {
    uint32_t v = def;
    (void)json_try_u32_obj(d, root, key, &v);
    return v;
}

static float config_f32(const json_doc *d, int root, const char *key, float def) {
    int tok = json_obj_get(d, root, key);
    if (tok < 0 || d->v[tok].type != JT_PRIMITIVE) return def;
    char tmp[64];
    int n = d->v[tok].end - d->v[tok].start;
    if (n <= 0 || n >= (int)sizeof(tmp)) return def;
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtof(tmp, NULL);
}

static void write_config_u32_array_or_default(FILE *fp, const json_doc *d, int root,
                                              const char *json_key, const char *gguf_key,
                                              const uint32_t *def, uint64_t def_n) {
    int arr = json_obj_get(d, root, json_key);
    if (arr < 0 || d->v[arr].type != JT_ARRAY) {
        write_kv_u32_array(fp, gguf_key, def, def_n);
        return;
    }
    uint32_t *vals = xmalloc((size_t)d->v[arr].size * sizeof(vals[0]));
    uint64_t n = 0;
    for (int i = arr + 1; i < d->len && d->v[i].parent == arr; i = json_skip(d, i)) {
        int64_t v = json_i64(d, i);
        vals[n++] = v < 0 ? 0u : (uint32_t)v;
    }
    write_kv_u32_array(fp, gguf_key, vals, n);
    free(vals);
}

static byte_buf write_hf_config_kv_raw(const char *hf_dir, int n_experts) {
    char *config_path = path_join(hf_dir, "config.json");
    size_t len = 0;
    char *text = read_file(config_path, &len);
    free(config_path);
    json_doc d = json_parse_text(text, len);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("bad config.json");
    const int root = 0;

    FILE *fp = tmpfile();
    if (!fp) die_errno("tmpfile", NULL);
    const uint32_t n_layer = config_u32(&d, root, "num_hidden_layers", 43);
    const uint32_t n_embd = config_u32(&d, root, "hidden_size", 4096);
    const uint32_t n_vocab = config_u32(&d, root, "vocab_size", 129280);
    const uint32_t n_head = config_u32(&d, root, "num_attention_heads", 64);
    const uint32_t n_head_kv = config_u32(&d, root, "num_key_value_heads", 1);
    const uint32_t head_dim = config_u32(&d, root, "head_dim", 512);
    const uint32_t n_rot = config_u32(&d, root, "qk_rope_head_dim", 64);
    const uint32_t n_lora_q = config_u32(&d, root, "q_lora_rank", 1024);
    const uint32_t n_lora_o = config_u32(&d, root, "o_lora_rank", 1024);
    const uint32_t n_out_group = config_u32(&d, root, "o_groups", 8);
    const uint32_t n_expert_used = config_u32(&d, root, "num_experts_per_tok", 6);
    const uint32_t n_ff_exp = config_u32(&d, root, "moe_intermediate_size", 2048);
    const uint32_t n_shared = config_u32(&d, root, "n_shared_experts", 1);
    const uint32_t n_hash = config_u32(&d, root, "num_hash_layers", 3);
    const uint32_t n_swa = config_u32(&d, root, "sliding_window", 128);
    const uint32_t n_index_head = config_u32(&d, root, "index_n_heads", 64);
    const uint32_t n_index_dim = config_u32(&d, root, "index_head_dim", 128);
    const uint32_t n_index_topk = config_u32(&d, root, "index_topk", 512);
    const uint32_t n_hc = config_u32(&d, root, "hc_mult", 4);
    const uint32_t n_hc_sink = config_u32(&d, root, "hc_sinkhorn_iters", 20);
    const uint64_t ctx = config_u32(&d, root, "max_position_embeddings", 1048576);
    const float rope_theta = config_f32(&d, root, "rope_theta", 10000.0f);
    const float compress_rope = config_f32(&d, root, "compress_rope_theta", 160000.0f);
    const float expert_scale = config_f32(&d, root, "routed_scaling_factor", 1.5f);
    const float rms_eps = config_f32(&d, root, "rms_norm_eps", 1e-6f);
    const float hc_eps = config_f32(&d, root, "hc_eps", 1e-6f);
    const float swiglu_limit = config_f32(&d, root, "swiglu_limit", 10.0f);

    write_kv_u32(fp, "general.alignment", DS4_GGUF_DEFAULT_ALIGNMENT);
    write_kv_string(fp, "general.name", "DeepSeek-V4-Flash-DSpark-Abliterated");
    write_kv_string(fp, "general.architecture", "deepseek4");
    write_kv_u32(fp, "deepseek4.block_count", n_layer);
    write_kv_u64(fp, "deepseek4.context_length", ctx);
    write_kv_u32(fp, "deepseek4.embedding_length", n_embd);
    write_kv_u32(fp, "deepseek4.vocab_size", n_vocab);
    write_kv_u32(fp, "deepseek4.attention.head_count", n_head);
    write_kv_u32(fp, "deepseek4.attention.head_count_kv", n_head_kv);
    write_kv_u32(fp, "deepseek4.attention.key_length", head_dim);
    write_kv_u32(fp, "deepseek4.attention.value_length", head_dim);
    write_kv_u32(fp, "deepseek4.rope.dimension_count", n_rot);
    write_kv_u32(fp, "deepseek4.attention.q_lora_rank", n_lora_q);
    write_kv_u32(fp, "deepseek4.attention.output_lora_rank", n_lora_o);
    write_kv_u32(fp, "deepseek4.attention.output_group_count", n_out_group);
    write_kv_u32(fp, "deepseek4.expert_count", (uint32_t)n_experts);
    write_kv_u32(fp, "deepseek4.expert_used_count", n_expert_used);
    write_kv_u32(fp, "deepseek4.expert_feed_forward_length", n_ff_exp);
    write_kv_u32(fp, "deepseek4.expert_shared_count", n_shared);
    write_kv_u32(fp, "deepseek4.hash_layer_count", n_hash);
    write_kv_u32(fp, "deepseek4.attention.sliding_window", n_swa);
    write_kv_u32(fp, "deepseek4.attention.indexer.head_count", n_index_head);
    write_kv_u32(fp, "deepseek4.attention.indexer.key_length", n_index_dim);
    write_kv_u32(fp, "deepseek4.attention.indexer.top_k", n_index_topk);
    write_kv_u32(fp, "deepseek4.hyper_connection.count", n_hc);
    write_kv_u32(fp, "deepseek4.hyper_connection.sinkhorn_iterations", n_hc_sink);
    write_kv_f32(fp, "deepseek4.rope.freq_base", rope_theta);
    write_kv_f32(fp, "deepseek4.rope.scaling.factor", config_f32(&d, root, "factor", 16.0f));
    int rope_scaling = json_obj_get(&d, root, "rope_scaling");
    write_kv_f32(fp, "deepseek4.rope.scaling.yarn_beta_fast", rope_scaling >= 0 ? config_f32(&d, rope_scaling, "beta_fast", 32.0f) : 32.0f);
    write_kv_f32(fp, "deepseek4.rope.scaling.yarn_beta_slow", rope_scaling >= 0 ? config_f32(&d, rope_scaling, "beta_slow", 1.0f) : 1.0f);
    write_kv_u64(fp, "deepseek4.rope.scaling.original_context_length",
                 rope_scaling >= 0 ? config_u32(&d, rope_scaling, "original_max_position_embeddings", 65536) : 65536);
    write_kv_f32(fp, "deepseek4.attention.compress_rope_freq_base", compress_rope);
    write_kv_f32(fp, "deepseek4.expert_weights_scale", expert_scale);
    write_kv_f32(fp, "deepseek4.attention.layer_norm_rms_epsilon", rms_eps);
    write_kv_f32(fp, "deepseek4.hyper_connection.epsilon", hc_eps);
    write_kv_bool(fp, "deepseek4.expert_weights_norm", true);
    uint32_t def_comp[46];
    for (uint32_t i = 0; i < 46; i++) def_comp[i] = (i < 2 || i >= 43) ? 0u : (i % 2 ? 128u : 4u);
    write_config_u32_array_or_default(fp, &d, root, "compress_ratios", "deepseek4.attention.compress_ratios", def_comp, 46);
    float swiglu[43];
    for (uint32_t i = 0; i < 43; i++) swiglu[i] = swiglu_limit;
    write_kv_f32_array(fp, "deepseek4.swiglu_clamp_exp", swiglu, 43);

    long end = ftell(fp);
    if (end < 0) die("ftell failed");
    byte_buf out = { .size = (size_t)end, .data = xmalloc((size_t)end) };
    rewind(fp);
    if (out.size && fread(out.data, 1, out.size, fp) != out.size) die("read tmp kv failed");
    fclose(fp);
    json_free(&d);
    free(text);
    return out;
}

static void write_imatrix_kvs(FILE *fp, const imatrix_store *im) {
    if (!imatrix_enabled(im)) return;
    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_FILE);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, im->file);

    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES);
    write_u32(fp, GGUF_TYPE_UINT64);
    write_u64(fp, (uint64_t)im->n_entries);

    if (im->dataset) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_DATASET);
        write_u32(fp, GGUF_TYPE_STRING);
        write_gguf_string(fp, im->dataset);
    }
    if (im->chunks > 0) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS);
        write_u32(fp, GGUF_TYPE_UINT64);
        write_u64(fp, (uint64_t)im->chunks);
    }
}

static gguf_file load_gguf_metadata(const char *path) {
    gguf_file g = {0};
    g.path = xstrdup(path);
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, "GGUF", 4) != 0) {
        die("bad GGUF template");
    }
    g.version = read_u32_le_fp(fp, "GGUF version");
    g.n_tensors = read_u64_le_fp(fp, "GGUF tensor count");
    g.n_kv = read_u64_le_fp(fp, "GGUF KV count");
    g.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;
    byte_span *kv_keep = xcalloc((size_t)g.n_kv, sizeof(kv_keep[0]));
    uint64_t n_kv_keep = 0;

    off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < g.n_kv; i++) {
        off_t rec_start = ftello(fp);
        if (rec_start < 0 || rec_start < kv_start) die("GGUF ftell failed");
        char *key = read_gguf_string_fp(fp);
        uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32_le_fp(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t n = read_u32_le_fp(fp, "GGUF expert count");
            if (n <= (uint32_t)INT_MAX) g.n_experts = (int)n;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT64) {
            uint64_t n = read_u64_le_fp(fp, "GGUF expert count");
            if (n <= (uint64_t)INT_MAX) g.n_experts = (int)n;
        } else {
            skip_gguf_value_fp(fp, type);
        }
        off_t rec_end = ftello(fp);
        if (rec_end < 0 || rec_end < rec_start) die("GGUF ftell failed");

        /*
         * Template GGUFs may already carry imatrix provenance from a previous
         * quantization.  Drop those keys and write the current run's keys later,
         * otherwise the output can contain duplicate GGUF metadata with stale
         * and new values.
         */
        if (!is_imatrix_kv_key(key)) {
            kv_keep[n_kv_keep++] = (byte_span){
                .start = (size_t)(rec_start - kv_start),
                .end = (size_t)(rec_end - kv_start),
            };
        }
        free(key);
    }
    off_t tensor_start = ftello(fp);
    if (tensor_start < 0 || tensor_start < kv_start) die("GGUF ftell failed");
    size_t kv_full_len = (size_t)(tensor_start - kv_start);
    uint8_t *kv_full = xmalloc(kv_full_len);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (kv_full_len && fread(kv_full, 1, kv_full_len, fp) != kv_full_len) die("GGUF KV read failed");

    for (uint64_t i = 0; i < n_kv_keep; i++) g.kv_raw_len += kv_keep[i].end - kv_keep[i].start;
    g.kv_raw = xmalloc(g.kv_raw_len);
    size_t kv_pos = 0;
    for (uint64_t i = 0; i < n_kv_keep; i++) {
        size_t n = kv_keep[i].end - kv_keep[i].start;
        memcpy(g.kv_raw + kv_pos, kv_full + kv_keep[i].start, n);
        kv_pos += n;
    }
    g.n_kv = n_kv_keep;
    free(kv_full);
    free(kv_keep);
    if (fseeko(fp, tensor_start, SEEK_SET) != 0) die("GGUF seek failed");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string_fp(fp);
        t->n_dims = (int)read_u32_le_fp(fp, "GGUF tensor rank");
        if (t->n_dims < 1 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = (int64_t)read_u64_le_fp(fp, "GGUF tensor dim");
        t->type = (ds4q_type)read_u32_le_fp(fp, "GGUF tensor type");
        t->old_offset = read_u64_le_fp(fp, "GGUF tensor offset");
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    off_t meta_end = ftello(fp);
    if (meta_end < 0) die("GGUF ftell failed");
    g.data_offset = ds4q_pad((size_t)meta_end, g.alignment);
    char **keys = xmalloc((size_t)g.n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) keys[i] = g.tensors[i].name;
    hmap_build(&g.tensor_map, keys, (int)g.n_tensors);
    free(keys);
    fclose(fp);
    return g;
}

static byte_buf read_gguf_tensor_data(const gguf_file *g, const char *path, const char *name) {
    int idx = hmap_get(&g->tensor_map, name);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in GGUF: %s\n", name);
        exit(1);
    }
    const tensor_meta *t = &g->tensors[idx];
    byte_buf b = { .size = t->size, .data = xmalloc(t->size) };
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    if (fseeko(fp, (off_t)(g->data_offset + t->old_offset), SEEK_SET) != 0) die_errno("seek GGUF", path);
    if (b.size && fread(b.data, 1, b.size, fp) != b.size) die_errno("read GGUF tensor", path);
    fclose(fp);
    return b;
}

static void output_context_append_tensor(output_context *out, const tensor_meta *src) {
    out->tensors = xrealloc(out->tensors, (size_t)(out->n_tensors + 1) * sizeof(out->tensors[0]));
    out->tensors[out->n_tensors++] = *src;
}

static bool output_context_has_tensor(const output_context *out, const char *name) {
    for (uint64_t i = 0; i < out->n_tensors; i++) {
        if (strcmp(out->tensors[i].name, name) == 0) return true;
    }
    return false;
}

static void output_context_plan_append(output_context *out,
                                       size_t *tensor_info,
                                       size_t *off,
                                       const tensor_meta *src,
                                       const char *name,
                                       ds4q_type target,
                                       size_t alignment) {
    tensor_meta t = *src;
    t.name = xstrdup(name);
    t.type = target;
    t.size = tensor_nbytes(target, t.ne, t.n_dims);
    t.new_offset = *off;
    *off += ds4q_pad(t.size, alignment);
    *tensor_info += gguf_string_size(t.name) + 4 + (size_t)t.n_dims * 8 + 4 + 8;
    output_context_append_tensor(out, &t);
}

static ds4q_type template_type_for_hf(const st_info *info) {
    if (strcmp(info->dtype, "I64") == 0) return DS4Q_TYPE_I32;
    if (strcmp(info->dtype, "F32") == 0) return DS4Q_TYPE_F32;
    if (strcmp(info->dtype, "F16") == 0) return DS4Q_TYPE_F16;
    if (strcmp(info->dtype, "BF16") == 0) return DS4Q_TYPE_BF16;
    return DS4Q_TYPE_F16;
}

static void tensor_meta_from_hf(st_db *db, const char *gguf_name, tensor_meta *t) {
    memset(t, 0, sizeof(*t));
    char *hf = NULL;
    if (parse_expert_tensor(gguf_name).is_expert) {
        expert_tensor e = parse_expert_tensor(gguf_name);
        char probe[320];
        const char *layer_prefix = str_starts(gguf_name, "mtp.") ? "mtp" : "layers";
        snprintf(probe, sizeof(probe), "%s.%d.ffn.experts.0.%s.weight",
                 layer_prefix, e.layer, expert_part_name(e.part));
        hf = xstrdup(probe);
    } else if (db_has(db, gguf_name)) {
        hf = xstrdup(gguf_name);
    } else {
        hf = hf_name_for_regular(gguf_name);
    }
    const st_info *info = db_info(db, hf);
    t->name = xstrdup(gguf_name);
    t->type = template_type_for_hf(info);
    if (parse_expert_tensor(gguf_name).is_expert) {
        if (info->n_dims != 2) die("expert tensor must be 2D");
        t->n_dims = 3;
        t->ne[0] = info->shape[1] * 2;
        t->ne[1] = info->shape[0];
        t->ne[2] = 256;
    } else if (strcmp(gguf_name, "mtp.2.confidence_head.proj.weight") == 0 &&
               info->n_dims == 2 && info->shape[0] == 1) {
        t->n_dims = 1;
        t->ne[0] = info->shape[1];
    } else if (g_dspark_source_mode && is_dspark_support_tensor(gguf_name) && info->n_dims == 1) {
        t->n_dims = 1;
        t->ne[0] = info->shape[0];
    } else {
        t->n_dims = info->n_dims;
        for (int i = 0; i < info->n_dims; i++) t->ne[i] = info->shape[info->n_dims - 1 - i];
    }
    t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    free(hf);
}

static void synthetic_template_append(gguf_file *tmpl, st_db *db, const char *name) {
    char *hf = NULL;
    if (parse_expert_tensor(name).is_expert) {
        expert_tensor e = parse_expert_tensor(name);
        const char *layer_prefix = str_starts(name, "mtp.") ? "mtp" : "layers";
        char probe[320];
        snprintf(probe, sizeof(probe), "%s.%d.ffn.experts.0.%s.weight",
                 layer_prefix, e.layer, expert_part_name(e.part));
        hf = xstrdup(probe);
    } else if (db_has(db, name)) {
        hf = xstrdup(name);
    } else {
        hf = hf_name_for_regular(name);
    }
    bool ok = db_has(db, hf);
    free(hf);
    if (!ok) return;
    tmpl->tensors = xrealloc(tmpl->tensors, (size_t)(tmpl->n_tensors + 1) * sizeof(tmpl->tensors[0]));
    tensor_meta_from_hf(db, name, &tmpl->tensors[tmpl->n_tensors]);
    tmpl->n_tensors++;
}

static gguf_file build_synthetic_template_from_hf(st_db *db, const char *hf_dir, int n_experts) {
    gguf_file tmpl = {0};
    tmpl.path = xstrdup("(hf-config)");
    tmpl.version = 3;
    tmpl.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;
    tmpl.n_experts = n_experts;
    byte_buf kv = write_hf_config_kv_raw(hf_dir, n_experts);
    tmpl.kv_raw = kv.data;
    tmpl.kv_raw_len = kv.size;
    tmpl.n_kv = 38;

    const char *top[] = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "output_hc_base.weight", "output_hc_fn.weight", "output_hc_scale.weight",
    };
    for (size_t i = 0; i < sizeof(top)/sizeof(top[0]); i++) synthetic_template_append(&tmpl, db, top[i]);
    for (int layer = 0; layer < 43; layer++) {
        char name[256];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer); synthetic_template_append(&tmpl, db, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer); synthetic_template_append(&tmpl, db, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer); synthetic_template_append(&tmpl, db, name);
        for (size_t i = 0; i < sizeof(layer_map)/sizeof(layer_map[0]); i++) {
            snprintf(name, sizeof(name), "blk.%d.%s", layer, layer_map[i].gguf);
            synthetic_template_append(&tmpl, db, name);
        }
    }
    if (g_dspark_source_mode) {
        for (int stage = 0; stage < 3; stage++) {
            char name[256];
            snprintf(name, sizeof(name), "mtp.%d.ffn_gate_exps.weight", stage); synthetic_template_append(&tmpl, db, name);
            snprintf(name, sizeof(name), "mtp.%d.ffn_down_exps.weight", stage); synthetic_template_append(&tmpl, db, name);
            snprintf(name, sizeof(name), "mtp.%d.ffn_up_exps.weight", stage); synthetic_template_append(&tmpl, db, name);
            for (size_t i = 0; i < sizeof(layer_map)/sizeof(layer_map[0]); i++) {
                snprintf(name, sizeof(name), "mtp.%d.%s", stage, layer_map[i].gguf);
                synthetic_template_append(&tmpl, db, name);
            }
        }
        const char *extra[] = {
            "mtp.2.norm.weight",
            "mtp.2.hc_head_base.weight",
            "mtp.2.hc_head_fn.weight",
            "mtp.2.hc_head_scale.weight",
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
            "mtp.2.confidence_head.proj.weight",
        };
        for (size_t i = 0; i < sizeof(extra)/sizeof(extra[0]); i++) synthetic_template_append(&tmpl, db, extra[i]);
    }
    for (uint64_t i = 0; i < tmpl.n_tensors; i++) tmpl.tensors[i].old_offset = 0;
    char **keys = xmalloc((size_t)tmpl.n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < tmpl.n_tensors; i++) keys[i] = tmpl.tensors[i].name;
    hmap_build(&tmpl.tensor_map, keys, (int)tmpl.n_tensors);
    free(keys);
    return tmpl;
}

static uint64_t fnv1a64_bytes(const uint8_t *data, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static output_context build_output_context(const st_db *db,
                                          const gguf_file *tmpl,
                                          const quant_policy *policy,
                                          const imatrix_store *im,
                                          const dspark_emit_metadata *dspark) {
    output_context out = {0};
    out.n_tensors = tmpl->n_tensors;
    out.n_kv_extra = extra_imatrix_kv_count(im);
    out.has_dspark_metadata = dspark && dspark->enabled;
    out.alignment = tmpl->alignment;
    out.tensors = xcalloc((size_t)out.n_tensors, sizeof(out.tensors[0]));
    size_t tensor_info = 0;
    size_t off = 0;
    for (uint64_t i = 0; i < out.n_tensors; i++) {
        const tensor_meta *src = &tmpl->tensors[i];
        tensor_meta *dst = &out.tensors[i];
        *dst = *src;
        dst->name = dspark_output_tensor_name(src->name);
        ds4q_type type = policy_type(policy, src->name, src);
        if (type == DS4Q_TYPE_COUNT) type = src->type;
        if (type != DS4Q_TYPE_I32 && !is_quantizable_target(type)) die("unsupported planned tensor type");
        if (ds4q_can_quantize(type) && src->ne[0] % ds4q_block_size(type) != 0) die("ne[0] not divisible by block size");
        dst->type = type;
        dst->size = tensor_nbytes(type, src->ne, src->n_dims);
        dst->new_offset = off;
        off += ds4q_pad(dst->size, tmpl->alignment);
        tensor_info += gguf_string_size(dst->name) + 4 + (size_t)dst->n_dims * 8 + 4 + 8;
    }

    if (dspark && dspark->enabled && db) {
        for (int stage = 0; stage < 3; stage++) {
            for (uint64_t ti = 0; ti < tmpl->n_tensors; ti++) {
                const tensor_meta *src = &tmpl->tensors[ti];
                if (!str_starts(src->name, "blk.0.")) continue;
                char name[512];
                snprintf(name, sizeof(name), "mtp.%d.%s", stage, src->name + strlen("blk.0."));
                if (!parse_expert_tensor(name).is_expert) {
                    char *hf_name = hf_name_for_regular(name);
                    bool has_tensor = db_has(db, hf_name);
                    free(hf_name);
                    if (!has_tensor) continue;
                }
                if (parse_expert_tensor(name).is_expert) {
                    char probe[320];
                    snprintf(probe, sizeof(probe), "mtp.%d.ffn.experts.0.%s.weight",
                             stage, expert_part_name(parse_expert_tensor(name).part));
                    if (!db_has(db, probe)) continue;
                }
                if (output_context_has_tensor(&out, name)) {
                    continue;
                }
                ds4q_type target = policy_type(policy, name, src);
                if (target == DS4Q_TYPE_COUNT) target = src->type;
                if (target != DS4Q_TYPE_I32 && !is_quantizable_target(target)) die("unsupported planned tensor type");
                if (ds4q_can_quantize(target) && src->ne[0] % ds4q_block_size(target) != 0) die("ne[0] not divisible by block size");
                output_context_plan_append(&out, &tensor_info, &off, src, name, target, tmpl->alignment);
            }
        }
        const struct {
            const char *name;
            ds4q_type type;
            int n_dims;
            int64_t ne[DS4Q_MAX_DIMS];
        } extra_specs[] = {
            { "mtp.2.norm.weight",                  DS4Q_TYPE_F32,  1, { 4096 } },
            { "mtp.2.hc_head_base.weight",          DS4Q_TYPE_F32,  1, { 4 } },
            { "mtp.2.hc_head_fn.weight",            DS4Q_TYPE_F16,  2, { 16384, 4 } },
            { "mtp.2.hc_head_scale.weight",         DS4Q_TYPE_F32,  1, { 1 } },
            { "mtp.0.main_proj.weight",             DS4Q_TYPE_F16,  2, { 12288, 4096 } },
            { "mtp.0.main_norm.weight",             DS4Q_TYPE_F32,  1, { 4096 } },
            { "mtp.2.markov_head.markov_w1.weight", DS4Q_TYPE_BF16, 2, { 256, 129280 } },
            { "mtp.2.markov_head.markov_w2.weight", DS4Q_TYPE_BF16, 2, { 256, 129280 } },
            { "mtp.2.confidence_head.proj.weight",  DS4Q_TYPE_BF16, 1, { 4352 } },
        };
        for (size_t i = 0; i < sizeof(extra_specs)/sizeof(extra_specs[0]); i++) {
            const char *name = extra_specs[i].name;
            if (!db_has(db, name)) continue;
            if (output_context_has_tensor(&out, name)) continue;
            tensor_meta t = {0};
            t.name = dspark_output_tensor_name(name);
            t.n_dims = extra_specs[i].n_dims;
            for (int j = 0; j < t.n_dims; j++) t.ne[j] = extra_specs[i].ne[j];
            t.type = extra_specs[i].type;
            ds4q_type target = t.type;
            if (target != DS4Q_TYPE_I32 && !is_quantizable_target(target)) die("unsupported planned tensor type");
            if (ds4q_can_quantize(target) && t.ne[0] % ds4q_block_size(target) != 0) die("ne[0] not divisible by block size");
            output_context_plan_append(&out, &tensor_info, &off, &t, t.name, target, tmpl->alignment);
        }
    }
    out.tensor_bytes = off;
    out.meta_size = 4 + 4 + 8 + 8 + tmpl->kv_raw_len +
                    extra_dspark_kv_size(dspark) +
                    extra_imatrix_kv_size(im) + tensor_info;
    out.data_offset = ds4q_pad(out.meta_size, tmpl->alignment);
    return out;
}

static void write_padding(FILE *fp, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

static void write_full_gguf(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                            const char *out_path, int n_experts, int n_threads,
                            const imatrix_store *imatrix,
                            const dspark_emit_metadata *dspark) {
    FILE *fp = fopen(out_path, "wb");
    if (!fp) die_errno("open output", out_path);
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, tmpl->version);
    write_u64(fp, out_ctx->n_tensors);
    write_u64(fp, tmpl->n_kv + out_ctx->n_kv_extra + extra_dspark_kv_count(dspark));
    if (fwrite(tmpl->kv_raw, 1, tmpl->kv_raw_len, fp) != tmpl->kv_raw_len) die("write GGUF KV failed");
    write_dspark_kvs(fp, dspark);
    write_imatrix_kvs(fp, imatrix);
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *t = &out_ctx->tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
        write_u32(fp, (uint32_t)t->type);
        write_u64(fp, t->new_offset);
    }
    long pos = ftell(fp);
    if (pos < 0) die("ftell failed");
    if ((size_t)pos > out_ctx->data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, out_ctx->data_offset - (size_t)pos);

    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *src = i < tmpl->n_tensors ? &tmpl->tensors[i] : &out_ctx->tensors[i];
        const tensor_meta *dst = &out_ctx->tensors[i];
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %s -> %s\n", i + 1, out_ctx->n_tensors, dst->name, ds4q_type_name(dst->type));
        byte_buf data = generate_tensor(db, dst->name, src, dst->type, n_experts, n_threads, imatrix);
        size_t expected = dst->size;
        if (data.size != expected) {
            fprintf(stderr, "error: generated size mismatch for %s: got %zu expected %zu\n", dst->name, data.size, expected);
            exit(1);
        }
        if (fwrite(data.data, 1, data.size, fp) != data.size) die_errno("write tensor", out_path);
        size_t padded = ds4q_pad(data.size, out_ctx->alignment);
        write_padding(fp, padded - data.size);
        fprintf(stderr, "       generated %.2f MiB\n", (double)data.size / 1048576.0);
        free(data.data);
    }
    fclose(fp);
}

static void patch_gguf_with_dspark(const gguf_file *src, const dspark_emit_metadata *dspark, const char *out_path) {
    if (!src) die("missing GGUF source");
    dspark_emit_metadata merged = {0};
    if (dspark && dspark->enabled) merged = *dspark;
    if (!merged.enabled) {
        merged.enabled = true;
    }
    if (merged.algorithm[0] == '\0') snprintf(merged.algorithm, sizeof(merged.algorithm), "%s", "dspark");
    if (merged.carrier_kind[0] == '\0') {
        snprintf(merged.carrier_kind, sizeof(merged.carrier_kind), "%s", guess_dspark_carrier_kind_from_gguf(src));
    }
    if (merged.model_id[0] == '\0') snprintf(merged.model_id, sizeof(merged.model_id), "%s", "deepseek-v4-flash-dspark");
    if (merged.backend_compat[0] == '\0') snprintf(merged.backend_compat, sizeof(merged.backend_compat), "%s", "cuda-spark");
    if (merged.version == 0) merged.version = 1;
    if (merged.n_mtp_layers == 0) merged.n_mtp_layers = 3;
    if (merged.block_size == 0) merged.block_size = 5;
    if (merged.noise_token_id == 0) merged.noise_token_id = 128799u;
    if (merged.markov_rank == 0) merged.markov_rank = 256u;
    if (merged.recommended_draft_tokens == 0) merged.recommended_draft_tokens = 2;
    if (merged.target_layer_ids[0] == 0 && merged.target_layer_ids[1] == 0 && merged.target_layer_ids[2] == 0) {
        merged.target_layer_ids[0] = 40u;
        merged.target_layer_ids[1] = 41u;
        merged.target_layer_ids[2] = 42u;
    }
    FILE *in = fopen(src->path, "rb");
    if (!in) die_errno("open source", src->path);
    FILE *out = fopen(out_path, "wb");
    if (!out) die_errno("open output", out_path);

    if (fwrite("GGUF", 1, 4, out) != 4) die("write GGUF magic failed");
    write_u32(out, src->version);
    write_u64(out, src->n_tensors);
    write_u64(out, src->n_kv + extra_dspark_kv_count(&merged));
    if (fwrite(src->kv_raw, 1, src->kv_raw_len, out) != src->kv_raw_len) die("write GGUF KV failed");
    write_dspark_kvs(out, &merged);
    for (uint64_t i = 0; i < src->n_tensors; i++) {
        const tensor_meta *t = &src->tensors[i];
        write_gguf_string(out, t->name);
        write_u32(out, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(out, (uint64_t)t->ne[j]);
        write_u32(out, (uint32_t)t->type);
        write_u64(out, t->old_offset);
    }

    long pos = ftell(out);
    if (pos < 0) die("ftell failed");
    write_padding(out, ds4q_pad((size_t)pos, src->alignment) - (size_t)pos);

    if (fseeko(in, (off_t)src->data_offset, SEEK_SET) != 0) die_errno("seek source", src->path);
    uint8_t buf[1 << 20];
    size_t nread;
    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, nread, out) != nread) die_errno("write patched GGUF", out_path);
    }
    if (ferror(in)) die_errno("read source GGUF", src->path);

    fclose(in);
    fclose(out);
}

static void print_plan(const gguf_file *tmpl, const output_context *out_ctx) {
    size_t tensor_bytes = 0;
    size_t changed = 0;
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        tensor_bytes += out_ctx->tensors[i].size;
        const tensor_meta *dst = &out_ctx->tensors[i];
        if (i < tmpl->n_tensors) {
            const tensor_meta *src = &tmpl->tensors[i];
            if (src->type != dst->type) {
                changed++;
                printf("type_change: %s %s -> %s\n", dst->name, ds4q_type_name(src->type), ds4q_type_name(dst->type));
            }
        } else {
            changed++;
            printf("type_change: %s <extra> -> %s\n", dst->name, ds4q_type_name(dst->type));
        }
    }
    printf("n_tensors: %" PRIu64 "\n", out_ctx->n_tensors);
    printf("meta_bytes: %zu\n", out_ctx->data_offset);
    printf("tensor_bytes_unpadded: %zu\n", tensor_bytes);
    printf("approx_file_bytes: %zu\n", out_ctx->data_offset + out_ctx->tensor_bytes);
    printf("type_changes: %zu\n", changed);
    if (out_ctx->has_dspark_metadata) {
        printf("dspark_metadata: enabled\n");
    }
}

/* =====
 * CLI
 */

typedef struct {
    char *hf_dir;
    char *template_gguf;
    char *patch_gguf;
    char *dspark_config_dir;
    char *out_gguf;
    char *compare_gguf;
    char *compare_tensor;
    char *imatrix_file;
    quant_policy policy;
    int n_experts;
    int n_threads;
    bool dry_run;
    bool overwrite;
    bool imatrix_strict;
    bool from_config;
} params;

static void usage(const char *argv0) {
    printf("usage: %s --hf DIR --from-config --out OUT.gguf [options]\n", argv0);
    printf("\nDeepSeek V4 Flash/Pro safetensors -> GGUF quantizer in plain C.\n\n");
    printf("patch mode:\n");
    printf("  --patch-gguf FILE      copy an existing GGUF and add DSpark metadata only\n");
    printf("  --dspark-config DIR    directory containing config.json for DSpark metadata\n");
    printf("\n");
    printf("options:\n");
    printf("  --hf DIR               Hugging Face model directory with model.safetensors.index.json\n");
    printf("  --from-config          derive DS4 GGUF metadata/tensor layout from HF config and safetensors\n");
    printf("  --template FILE        explicitly use existing DS4 GGUF for metadata, tensor order, shapes\n");
    printf("  --out FILE             output GGUF path\n");
    printf("  --compare-gguf FILE    reference GGUF for --compare-tensor, default template\n");
    printf("  --compare-tensor NAME  regenerate one tensor, byte-compare, and exit\n");
    printf("  --overwrite            replace --out if it already exists\n");
    printf("  --dry-run              print output plan without reading HF tensor data\n");
    printf("  --imatrix FILE         legacy .dat imatrix from ds4 --imatrix-out\n");
    printf("  --imatrix-strict       fail if a quantized tensor has no matching imatrix vector\n");
    printf("  --experts TYPE         set routed w1/w2/w3 expert tensors to TYPE\n");
    printf("  --routed-w1 TYPE       routed gate expert tensor type\n");
    printf("  --routed-w2 TYPE       routed down expert tensor type\n");
    printf("  --routed-w3 TYPE       routed up expert tensor type\n");
    printf("  --attention-proj TYPE  attn_q/kv/output projection type\n");
    printf("  --attention TYPE       other 2D attention/indexer/compressor type\n");
    printf("  --shared TYPE          shared expert tensor type\n");
    printf("  --embedding TYPE       token embedding type\n");
    printf("  --output TYPE          output.* tensor type\n");
    printf("  --dense TYPE           remaining 2D+ non-routed tensor type\n");
    printf("  --tensor-type PFX=TYPE exact tensor-name or prefix override; may repeat\n");
    printf("  --n-experts N          routed expert count, default template metadata\n");
    printf("  --threads N            expert worker count, default 8\n");
    printf("\nTYPE examples: f16, f32, bf16, q8_0, q4_k, q2_k, iq2_xxs\n");
}

static char *need_value(int argc, char **argv, int *i, const char *arg) {
    if (++*i >= argc) {
        fprintf(stderr, "error: missing value for %s\n", arg);
        exit(1);
    }
    return argv[*i];
}

static bool file_exists(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static params parse_args(int argc, char **argv) {
    params p = {0};
    p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = DS4Q_TYPE_COUNT;
    p.policy.attention_proj = p.policy.attention = p.policy.shared = DS4Q_TYPE_COUNT;
    p.policy.embedding = p.policy.output = p.policy.dense = DS4Q_TYPE_COUNT;
    p.n_experts = 0;
    p.n_threads = 8;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--hf") == 0) {
            p.hf_dir = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--template") == 0) {
            p.template_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--from-config") == 0) {
            p.from_config = true;
        } else if (strcmp(arg, "--patch-gguf") == 0) {
            p.patch_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--dspark-config") == 0) {
            p.dspark_config_dir = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--out") == 0) {
            p.out_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-gguf") == 0) {
            p.compare_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-tensor") == 0) {
            p.compare_tensor = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--overwrite") == 0) {
            p.overwrite = true;
        } else if (strcmp(arg, "--dry-run") == 0) {
            p.dry_run = true;
        } else if (strcmp(arg, "--imatrix") == 0) {
            p.imatrix_file = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-strict") == 0) {
            p.imatrix_strict = true;
        } else if (strcmp(arg, "--experts") == 0 || strcmp(arg, "--routed") == 0) {
            ds4q_type t = parse_type(need_value(argc, argv, &i, arg));
            p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = t;
        } else if (strcmp(arg, "--routed-w1") == 0 || strcmp(arg, "--routed-gate") == 0) {
            p.policy.routed_w1 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w2") == 0 || strcmp(arg, "--routed-down") == 0) {
            p.policy.routed_w2 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w3") == 0 || strcmp(arg, "--routed-up") == 0) {
            p.policy.routed_w3 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention-proj") == 0 || strcmp(arg, "--attn-proj") == 0) {
            p.policy.attention_proj = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention") == 0) {
            p.policy.attention = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--shared") == 0) {
            p.policy.shared = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--embedding") == 0) {
            p.policy.embedding = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--output") == 0) {
            p.policy.output = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--dense") == 0) {
            p.policy.dense = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--tensor-type") == 0) {
            char *spec = need_value(argc, argv, &i, arg);
            char *eq = strchr(spec, '=');
            if (!eq || eq == spec || !eq[1]) die("bad --tensor-type, expected NAME=TYPE");
            *eq = '\0';
            p.policy.overrides = xrealloc(p.policy.overrides, (size_t)(p.policy.n_overrides + 1) * sizeof(p.policy.overrides[0]));
            p.policy.overrides[p.policy.n_overrides++] = (type_override){ xstrdup(spec), parse_type(eq + 1) };
        } else if (strcmp(arg, "--n-experts") == 0) {
            p.n_experts = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--threads") == 0) {
            p.n_threads = atoi(need_value(argc, argv, &i, arg));
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            exit(1);
        }
    }
    const bool patch_mode = p.patch_gguf != NULL;
    if (patch_mode) {
        if (!p.dspark_config_dir) die("--dspark-config is required with --patch-gguf");
    } else {
        if (!p.hf_dir) die("--hf is required");
        if (!p.from_config && !p.template_gguf) die("--from-config is required unless --template is explicitly provided");
        if (p.from_config && p.template_gguf) die("--from-config and --template are mutually exclusive");
    }
    if (!p.dry_run && !p.compare_tensor && !p.out_gguf) die("--out is required unless --dry-run or --compare-tensor is used");
    if (p.compare_tensor && !p.compare_gguf) p.compare_gguf = p.template_gguf;
    if (p.out_gguf && file_exists(p.out_gguf) && !p.overwrite) die("output exists; use --overwrite");
    return p;
}

static void free_gguf_file(gguf_file *g) {
    free(g->path);
    free(g->kv_raw);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    hmap_free(&g->tensor_map);
    memset(g, 0, sizeof(*g));
}

static void compare_one_tensor(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                               const params *p, const imatrix_store *imatrix) {
    int idx = hmap_get(&tmpl->tensor_map, p->compare_tensor);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in template: %s\n", p->compare_tensor);
        exit(1);
    }
    fprintf(stderr, "regenerating %s as %s\n",
            p->compare_tensor, ds4q_type_name(out_ctx->tensors[idx].type));
    byte_buf generated = generate_tensor(db, p->compare_tensor, &tmpl->tensors[idx],
                                         out_ctx->tensors[idx].type, p->n_experts, p->n_threads, imatrix);
    gguf_file ref = load_gguf_metadata(p->compare_gguf);
    byte_buf reference = read_gguf_tensor_data(&ref, p->compare_gguf, p->compare_tensor);
    printf("tensor: %s\n", p->compare_tensor);
    printf("type: %s\n", ds4q_type_name(out_ctx->tensors[idx].type));
    printf("generated_bytes: %zu\n", generated.size);
    printf("reference_bytes: %zu\n", reference.size);
    printf("generated_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(generated.data, generated.size));
    printf("reference_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(reference.data, reference.size));
    size_t mismatches = 0;
    size_t first = SIZE_MAX;
    const size_t n = generated.size < reference.size ? generated.size : reference.size;
    for (size_t i = 0; i < n; i++) {
        if (generated.data[i] != reference.data[i]) {
            if (first == SIZE_MAX) first = i;
            mismatches++;
        }
    }
    if (generated.size != reference.size) {
        if (first == SIZE_MAX) first = n;
        mismatches += generated.size > reference.size ? generated.size - reference.size : reference.size - generated.size;
    }
    if (!mismatches) {
        printf("byte_compare: OK\n");
    } else {
        printf("byte_compare: FAIL mismatches=%zu first=%zu\n", mismatches, first);
    }
    free(generated.data);
    free(reference.data);
    free_gguf_file(&ref);
}

int main(int argc, char **argv) {
    params p = parse_args(argc, argv);
    if (p.patch_gguf) {
        gguf_file src = load_gguf_metadata(p.patch_gguf);
        dspark_emit_metadata dspark = {0};
        if (!load_dspark_emit_metadata(p.dspark_config_dir, &dspark)) {
            dspark_emit_metadata_set_defaults(&dspark, guess_dspark_carrier_kind_from_gguf(&src));
        } else if (dspark.algorithm[0] == '\0') {
            snprintf(dspark.algorithm, sizeof(dspark.algorithm), "%s", "dspark");
            if (dspark.carrier_kind[0] == '\0') {
                snprintf(dspark.carrier_kind, sizeof(dspark.carrier_kind), "%s", guess_dspark_carrier_kind_from_gguf(&src));
            }
        }
        dspark.enabled = true;
        patch_gguf_with_dspark(&src, &dspark, p.out_gguf);
        fprintf(stderr, "wrote %s\n", p.out_gguf);
        free_gguf_file(&src);
        return 0;
    }

    imatrix_store imatrix = {0};
    if (p.imatrix_file) imatrix_load(&imatrix, p.imatrix_file, p.imatrix_strict);

    st_db db;
    db_open(&db, p.hf_dir);
    dspark_emit_metadata dspark = {0};
    if (db_has(&db, "mtp.0.main_proj.weight") &&
        db_has(&db, "mtp.2.markov_head.markov_w1.weight")) {
        g_dspark_source_mode = true;
        if (!load_dspark_emit_metadata(p.hf_dir, &dspark)) {
            dspark_emit_metadata_set_defaults(&dspark, "official-dspark");
        } else {
            if (dspark.algorithm[0] == '\0') snprintf(dspark.algorithm, sizeof(dspark.algorithm), "%s", "dspark");
            if (dspark.carrier_kind[0] == '\0') snprintf(dspark.carrier_kind, sizeof(dspark.carrier_kind), "%s", "official-dspark");
        }
        dspark.enabled = true;
    } else if (db_has(&db, "mtp.0.e_proj.weight")) {
        g_dspark_source_mode = false;
        if (!load_dspark_emit_metadata(p.hf_dir, &dspark)) {
            dspark_emit_metadata_set_defaults(&dspark, "legacy-mtp");
        } else {
            if (dspark.algorithm[0] == '\0') snprintf(dspark.algorithm, sizeof(dspark.algorithm), "%s", "dspark");
            if (dspark.carrier_kind[0] == '\0') snprintf(dspark.carrier_kind, sizeof(dspark.carrier_kind), "%s", "legacy-mtp");
        }
        dspark.enabled = true;
    } else {
        g_dspark_source_mode = false;
    }
    if (p.n_experts <= 0) {
        p.n_experts = 256;
        fprintf(stderr, "using %d routed experts from HF/config default\n", p.n_experts);
    } else {
        fprintf(stderr, "using %d routed experts from --n-experts\n", p.n_experts);
    }
    gguf_file tmpl = {0};
    if (p.from_config) {
        fprintf(stderr, "building GGUF layout from HF config/safetensors (no GGUF template)\n");
        tmpl = build_synthetic_template_from_hf(&db, p.hf_dir, p.n_experts);
        g_template_file = NULL;
        g_template_path = NULL;
    } else {
        tmpl = load_gguf_metadata(p.template_gguf);
        g_template_file = &tmpl;
        g_template_path = tmpl.path;
        if (tmpl.n_experts > 0 && p.n_experts == 256) {
            p.n_experts = tmpl.n_experts;
            fprintf(stderr, "using %d routed experts from template metadata\n", p.n_experts);
        }
    }
    output_context out_ctx = build_output_context(&db, &tmpl, &p.policy, &imatrix, &dspark);
    print_plan(&tmpl, &out_ctx);
    if (p.dry_run) {
        db_close(&db);
        imatrix_free(&imatrix);
        free_gguf_file(&tmpl);
        free(out_ctx.tensors);
        for (int i = 0; i < p.policy.n_overrides; i++) free(p.policy.overrides[i].prefix);
        free(p.policy.overrides);
        return 0;
    }

    if (p.compare_tensor) {
        compare_one_tensor(&db, &tmpl, &out_ctx, &p, &imatrix);
        db_close(&db);
        imatrix_free(&imatrix);
        free_gguf_file(&tmpl);
        free(out_ctx.tensors);
        return 0;
    }
    write_full_gguf(&db, &tmpl, &out_ctx, p.out_gguf, p.n_experts, p.n_threads, &imatrix, &dspark);
    fprintf(stderr, "wrote %s\n", p.out_gguf);

    db_close(&db);
    imatrix_free(&imatrix);
    free_gguf_file(&tmpl);
    free(out_ctx.tensors);
    for (int i = 0; i < p.policy.n_overrides; i++) free(p.policy.overrides[i].prefix);
    free(p.policy.overrides);
    return 0;
}
