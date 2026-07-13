import assert from "node:assert/strict"
import { total } from "./math.js"

assert.equal(total([2, 3, 5]), 10)
assert.equal(total([]), 0)
