# DS4 · DSpark GB10

Fork di [antirez/ds4](https://github.com/antirez/ds4) ottimizzato per
NVIDIA DGX Spark / GB10 (`sm_121`) e per il modello
**DeepSeek V4 Flash DSpark Abliterated Q2**.

> **Stato:** operativo su GB10. Il profilo DSpark fast raggiunge circa
> **17.6 tok/s generati e 18.1 tok/s effettivi** nel test coding; il risultato
> dipende da contesto, temperatura, prompt e acceptance.

## Indice

- [Cosa aggiunge il fork](#cosa-aggiunge-il-fork)
- [Requisiti](#requisiti)
- [Build](#build)
- [Modello DSpark](#modello-dspark)
- [Avvio rapido](#avvio-rapido)
- [Configurazioni consigliate](#configurazioni-consigliate)
- [Opzioni CLI](#opzioni-cli)
- [Prestazioni](#prestazioni)
- [Server HTTP e agent](#server-http-e-agent)
- [Servizio systemd](#servizio-systemd)
- [KV cache e SSD streaming](#kv-cache-e-ssd-streaming)
- [Diagnostica e test](#diagnostica-e-test)
- [Struttura delle modifiche](#struttura-delle-modifiche)

## Cosa aggiunge il fork

| Area | Implementazione |
|---|---|
| Backend | CUDA GB10 con PTX JIT; il percorso validato evita `CUDA_ARCH=sm_121` esplicito |
| DSpark | Worker e tensori DSpark embedded nel modello base; non richiede un secondo modello piccolo |
| Speculative decode | Pipeline ufficiale `main_hidden → main_proj → stages → Markov → confidence`, con verifica target combinata N=2/N=3 |
| Target forward | Attention non-causale DSpark, output-head microbatch, cache HC e hot target cache |
| CUDA | Q8 share-warp, graph decode, prewarm Q8→F16, cache HBM dei pesi caldi |
| KV | KV cache F16 per contesti grandi, checkpoint persistenti e riuso dei prefissi |
| Server | API OpenAI/Responses/Anthropic, tool call DSML, replay esatto dei tool e continuazioni KV |
| Agent | `ds4-agent` con percorso speculative decode integrato |
| Tooling | Benchmark, profiler, diagnostica acceptance e test di identità token |

### DSpark non è MTP legacy

`--mtp` carica un draft model esterno. DSpark invece è contenuto nel GGUF base:
si usa `--dspark` senza dichiarare un secondo modello.

L’abliteration è un requisito del checkpoint scelto, non una modifica runtime:
il programma deve solo caricare il modello DSpark abliterated convertito.

## Requisiti

- DGX Spark / GB10 con CUDA funzionante e memoria unificata sufficiente.
- Linux ARM64, `nvcc`, `gcc`, `make` e `curl`.
- Circa 87 GiB per il GGUF Q2 DSpark, oltre a KV, cache e sistema.
- Per il profilo prestazionale: nessun altro carico GPU e SSD streaming esplicito disattivato.

## Build

```sh
make cuda-spark                 # build consigliata su GB10
make -j$(nproc) ds4-server      # server HTTP
make test                       # test completi
./ds4 --help
./ds4-server --help
```

Il target GB10 usa PTX JIT. Evitare di forzare `CUDA_ARCH=sm_121` se si vuole
replicare il benchmark validato.

## Modello DSpark

Checkpoint sorgente supportato:

```text
Valent1qw/DeepSeek-V4-Flash-DSpark-Abliterated
```

Il checkpoint contiene safetensors, configurazione, tokenizer e tensori worker
DSpark. Download, conversione e quantizzazione sono separati dal codice:

```sh
./download_model.sh dspark-source
./quantize_dspark.sh \
  --out gguf/DeepSeek-V4-Flash-DSpark-Abliterated-COMBINED-Q2.correct.gguf
```

Il file usato dal fork pubblicato è:

```text
gguf/DeepSeek-V4-Flash-DSpark-Abliterated-COMBINED-Q2.correct.gguf
```

Verifica del modello:

```sh
./ds4 --inspect \
  --model gguf/DeepSeek-V4-Flash-DSpark-Abliterated-COMBINED-Q2.correct.gguf \
  --cuda --dspark
```

Nei log devono comparire `algorithm=dspark`, `carrier=official-dspark` e
`official DSpark speculative runtime enabled`.

## Avvio rapido

### Chat greedy

```sh
./ds4 --cuda \
  --model gguf/DeepSeek-V4-Flash-DSpark-Abliterated-COMBINED-Q2.correct.gguf \
  --dspark --ctx 131072 --tokens 32768 -t 10 \
  --prefill-chunk 2048 --temp 0 --nothink
```

### Profilo GB10 da 17–18 tok/s

```sh
tools/perf/dspark/run-17tps.sh \
  tests/test-vectors/prompts/long_code_audit.txt
```

Il launcher imposta il GGUF corretto, `DS4_CUDA_FAST_VERIFY=1`, il percorso
MoE deterministico e tutti i parametri del benchmark.

## Configurazioni consigliate

### Qualità e identità token

Usare il verifier deterministico, senza fast mode:

```sh
env DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
  ./ds4 --cuda --model "$MODEL" --dspark \
  --ctx 131072 --tokens 32768 -t 10 --prefill-chunk 2048 \
  --temp 0 --nothink
```

È il profilo per confronti token-per-token e regressioni numeriche.

### Throughput DSpark

```sh
env DS4_CUDA_FAST_VERIFY=1 \
    DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
    DS4_GRAPH_DECODE=1 \
  ./ds4 --cuda --model "$MODEL" --dspark \
  --ctx 131072 --tokens 32768 -t 10 --prefill-chunk 2048 \
  --temp 0 --nothink
```

`FAST_VERIFY` accelera attention batched, GEMM e ordinamento delle righe.
È self-consistent e adatto al lavoro quotidiano, ma il programma segnala che
può produrre una sequenza greedy diversa dal target deterministico canonico.

### Parametri da evitare nel benchmark

- `--mtp`: non serve con il GGUF DSpark embedded e cambia percorso.
- `--ssd-streaming`: modalità di capacità, non di prestazioni quando il modello entra nella memoria unificata.
- `--quality`: utile solo per debug numerico; riduce sensibilmente il throughput.
- `--think-max`: aumenta i token di ragionamento, quindi il tempo alla risposta finale.

## Opzioni CLI

| Opzione | Funzione |
|---|---|
| `--model FILE`, `-m FILE` | GGUF da caricare |
| `--cuda`, `--cpu`, `--metal` | Seleziona backend |
| `--dspark [FILE]` | Abilita DSpark embedded; `FILE` resta per GGUF split legacy |
| `--ctx N`, `-c N` | Capacità del contesto/KV |
| `--tokens N`, `-n N` | Massimo token generati |
| `-t N` | Thread CPU helper |
| `--prefill-chunk N` | Dimensione chunk prefill; `2048` è il valore GB10 consigliato |
| `--temp F` | Temperatura; `0` = greedy |
| `--nothink` | Disabilita il ragionamento esteso |
| `--think`, `--think-max` | Abilita ragionamento normale/massimo |
| `--warm-weights` | Precarica i pesi densi caldi |
| `--power N` | Limite duty-cycle GPU; `85` può aiutare sessioni sostenute |
| `--ssd-streaming` | Abilita streaming da SSD per modelli che non entrano in RAM |
| `--kv-disk-dir DIR` | Directory checkpoint KV persistenti |
| `--dump-tokens` | Stampa tokenizzazione e termina |
| `--dump-logits FILE` | Salva logits completi |
| `--dump-logprobs FILE` | Salva alternative greedy |
| `--inspect` | Mostra metadati e termina |

## Prestazioni

### Benchmark isolato DSpark su GB10

Modello Q2 corretto, `ctx=131072`, `-t 10`, chunk `2048`, greedy, prefill
escluso dal decode:

| Profilo | Decode | Effective | Combined | Note |
|---|---:|---:|---:|---|
| Safe verify | 14.53 tok/s | 14.82 tok/s | 96.3% | deterministico |
| Fast verify | **17.59 tok/s** | **18.11 tok/s** | 94.6% | profilo throughput |

Fast verify ha prodotto circa **+21.1%** sul decode misurato. Acceptance del
test fast: `p0=0.854`, `p1=0.752`, profondità media circa `1.98`.

### Sessione reale Pi Code

Su una cronologia agent coding arrivata a circa 48k token:

| Fase | Misura |
|---|---:|
| Prefill | ~249 tok/s |
| Decode osservato | 13.7–14.0 tok/s |
| GPU | ~93% |
| GPU memory | ~27.6 GiB |
| Temperatura | ~74 °C |

Il calo rispetto a 18.11 tok/s è atteso: l’attenzione KV cresce con il contesto
reale e il risultato isolato non rappresenta ogni posizione da 48k/131k.

### Matrice storica GB10

Decode steady-state con fast verify, MTP storico e contesti 4k–32k:

| Contesto | Plain greedy | DSpark/MTP greedy | DSpark/MTP sampled |
|---:|---:|---:|---:|
| 4k | 12.7 | 21.3 | 18.8 |
| 8k | 12.6 | 21.2 | 18.8 |
| 16k | 12.4 | 20.7 | 18.9 |
| 32k | 11.6 | 19.9 | 17.9 |

I valori sono riferimenti di laboratorio; per il modello DSpark abliterated e
workload Pi Code usare il profilo e le misure della tabella precedente.

### Misurare il proprio carico

```sh
tools/perf/dspark/run-17tps.sh PROMPT.txt
journalctl -u ds4-server.service -f
nvidia-smi dmon -s pucmt
```

Il server stampa `prefill`, `gen`, `avg`, `prompt done` e `finish`. Il prefill
non va mescolato al decode quando si confrontano i tok/s.

## Server HTTP e agent

Avvio manuale:

```sh
./ds4-server --cuda --dspark \
  --model gguf/DeepSeek-V4-Flash-DSpark-Abliterated-COMBINED-Q2.correct.gguf \
  --ctx 131072 --tokens 32768 -t 10 --prefill-chunk 2048 \
  --host 0.0.0.0 --port 8000 --cors --warm-weights
```

Endpoint disponibili:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
POST /v1/completions
POST /v1/messages
```

Test rapido:

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4-flash","temperature":0,"max_tokens":128,"messages":[{"role":"user","content":"Spiega e correggi questo bug C: strcpy(dst, src)."}]}'
```

Sono supportati streaming SSE, tool call DSML, tool OpenAI/Anthropic e
continuazioni compatibili con Pi Code/Codex.

## Servizio systemd

La unità pubblicata è [deploy/ds4-server.service](deploy/ds4-server.service).
Installa e aggiorna il servizio:

```sh
sudo install -m 0644 deploy/ds4-server.service /etc/systemd/system/ds4-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now ds4-server.service
sudo systemctl restart ds4-server.service
systemctl status ds4-server.service
```

Il servizio usa il GGUF DSpark corretto, `--dspark`, `ctx=131072`, chunk 2048,
fast verify, MoE deterministico e KV disk cache. Log:

```sh
journalctl -u ds4-server.service -f
```

## KV cache e SSD streaming

La KV cache persistente accelera prefissi ripetuti e continuazioni agent:

```text
--kv-disk-dir /path/kv-cache
--kv-disk-space-mb 98304
--kv-cache-min-tokens 512
--kv-cache-cold-max-tokens 60000
--kv-cache-continued-interval-tokens 8192
--kv-cache-boundary-trim-tokens 32
--kv-cache-boundary-align-tokens 2048
```

La cache KV non aumenta il decode di un prompt nuovo; riduce soprattutto il
tempo di prefill quando il prefisso è riutilizzabile.

SSD streaming va usato solo quando il modello non entra nella memoria
disponibile. Per il benchmark GB10 full-resident deve restare disattivato:
ogni miss expert introduce latenza e può falsare il throughput.

## Diagnostica e test

```sh
make test
./ds4_test --dspark-runtime
./ds4_test --server
./ds4 --dump-tokens -p 'test prompt'
DS4_DSPARK_TIMING=1 DS4_DSPARK_SPEC_LOG=1 ./ds4 ...
```

Variabili utili:

| Variabile | Uso |
|---|---|
| `DS4_CUDA_FAST_VERIFY=1` | Verifier CUDA veloce; throughput, possibile differenza greedy |
| `DS4_CUDA_MOE_NO_ATOMIC_DOWN=1` | Riduzione MoE deterministica |
| `DS4_GRAPH_DECODE=1` | CUDA graph decode |
| `DS4_DSPARK_TIMING=1` | Tempi per iterazione DSpark |
| `DS4_DSPARK_SPEC_LOG=1` | Draft, commit e fallback |
| `DS4_DSPARK_NO_COST_ADAPTIVE=1` | Disabilita scheduler adattivo |
| `DS4_CUDA_NO_HBM_CACHE=1` | A/B senza cache pesi HBM; solo debug |
| `DS4_CUDA_HOT_TARGET_CACHE_MB=N` | Budget cache target |

Quando si modifica CUDA, verificare sempre token, acceptance e qualità su
coding, chat, tool call e long context. Non considerare sufficiente un aumento
del singolo kernel se peggiora l’effective tok/s o la risposta.

## Struttura delle modifiche

- `ds4.c` — sessione, scheduler DSpark, combined verifier, KV e server wiring.
- `ds4_cuda.cu` — kernel CUDA, attention DSpark, output-head, MoE e cache HBM.
- `ds4_gpu.h` — interfacce backend CUDA.
- `ds4_dspark_runtime.c/.h` — metadati, gating e utilità DSpark.
- `tools/perf/dspark/` — launcher benchmark DSpark.
- `deploy/ds4-server.service` — servizio systemd GB10.
- `FORK_RELEASE.md` — delta rispetto all’upstream e note di pubblicazione.

## Licenza e upstream

Il progetto mantiene la licenza e gli avvisi dell’upstream DS4. Le modifiche
specifiche del fork sono pensate per essere confrontate e, dove applicabile,
proposte a monte. Vedere `git log`, `FORK_RELEASE.md` e i remote Git configurati
per distinguere il fork da `upstream`.
