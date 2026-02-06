# Phase 2: The Solana Predator — Design Document

**Goal:** Apply the proven Spore/Swarm learning rule to real-world Solana market data. Predict price direction from raw binary account state.

**Architecture:** Scanning Swarm of 200 Spores (8-4-1 each), each looking at a different byte of the raw Raydium pool state. Evolutionary feature selection: Spores on price-relevant bytes converge, Spores on junk die. Oracle provides reward from parsed price lookahead.

**Key Insight:** The learning rule (Hebbian + signed dopamine + eligibility traces) is universal. The topology (8-4-1) is universal. Only the *wiring* changes: instead of 8 Spores looking at 8 known bits, 200 Spores look at 200 unknown bytes of real binary data.

---

## Component 1: `solana-sniffer` (Data Capture)

**Binary:** `solana-sniffer/src/main.rs` (standalone Rust crate, separate from spore-sim)

**Purpose:** Connect to Helius Yellowstone gRPC, subscribe to Raydium pool account updates, dump raw bytes to disk.

**Connection:**
- Helius Yellowstone gRPC (Geyser plugin)
- Requires Helius API key (passed as `--api-key` or `HELIUS_API_KEY` env var)
- Subscribe to `AccountUpdate` for one pool address

**CLI:**
```
solana-sniffer --pool 58oQChX4yWmvJFWGf6JuSCniYcxPdGWoTcue7CKRfyuY \
               --api-key <HELIUS_KEY> \
               --output market_vibrations.bin
```

**File Format:** `market_vibrations.bin`
- Sequential stream of fixed-size records
- Each record: `[slot: u64 LE (8 bytes)][data: 1024 bytes (zero-padded)]`
- Total per record: 1032 bytes
- No headers, no separators, no JSON
- Raw account data truncated or zero-padded to exactly 1024 bytes

**Default Pool:** Raydium SOL/USDC v4 AMM: `58oQChX4yWmvJFWGf6JuSCniYcxPdGWoTcue7CKRfyuY`

---

## Component 2: `solana_env.rs` (The Dojo)

**Module:** `spore-sim/src/solana_env.rs`

**Purpose:** Read `market_vibrations.bin`, feed raw bytes to Spores, compute Oracle reward.

**Data Flow:**
1. **Read:** Load .bin file, iterate through 1032-byte records
2. **Sensory Mapping:** Each Spore `i` is assigned a byte offset `offset_i` (0-1023). It reads `data[offset_i]` and converts it to 8 input bits (bit 0 = LSB, bit 7 = MSB), each as f32 (0.0 or 1.0).
3. **Oracle (Auto-Labeling):**
   - Extract `pool_coin_amount` (u64 LE) and `pool_pc_amount` (u64 LE) from known Borsh offsets in the raw data
   - Compute price = pc_amount / coin_amount (USDC per SOL)
   - Look ahead N records (configurable, default 100)
   - If price[T+N] > price[T]: reward = +1.0 (Buy was correct)
   - If price[T+N] <= price[T]: reward = -cortisol (Buy was wrong)
4. **Goal:** Spore outputs 1 (Buy) if future price > current price

**Teacher/Student Asymmetry:**
- Oracle (teacher): Parses price from known Borsh schema offsets
- Spores (students): See raw bits, discover which bytes correlate with reward
- Once trained, Oracle is no longer needed. The converged Spore drives predictions.

---

## Component 3: Scanning Swarm

**Architecture:** 200 Spores (8-4-1 each), all targeting 1 output bit: Buy (1) or Don't Buy (0).

**Wiring:**
- Each Spore is assigned a random byte offset in [0, 1023]
- Some offsets may overlap (redundancy on interesting bytes)
- Some offsets will hit junk (padding, discriminators, pubkeys)

**Selection Pressure:**
- Spores on price-relevant bytes find correlation with Oracle reward -> converge (>55% accuracy)
- Spores on junk bytes find no correlation -> stay at chance (~50%) -> get rejuvenated
- After training: inspect converged Spores' byte offsets to discover where the signal lives

**Consensus:** Best-expert-wins (same as mirror task). The Spore with highest `recent_accuracy` drives the Buy signal.

---

## Component 4: main.rs Integration

**Flag:** `cargo run --release -- --solana --data market_vibrations.bin`

**Behavior:**
1. Load market_vibrations.bin
2. Initialize Scanning Swarm (200 Spores, random byte offsets)
3. Use proven hyperparameters from Phase 1 (learning_rate, trace_decay, etc.)
4. Run training loop: feed records, compute Oracle reward, learn
5. Report: which Spore(s) converged, at what byte offsets, with what accuracy
6. Success criterion: any Spore > 55% accuracy on price direction prediction

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data source | Helius Yellowstone gRPC | High-fidelity, matches live battlefield latency |
| File format | [slot u64][data 1024B] | Compact, raw, aligned to memory pages |
| Spore architecture | 8-4-1 (unchanged) | Universal DNA — only wiring changes |
| Input mapping | 1 byte -> 8 bits per Spore | Preserves INPUT_SIZE=8 |
| Number of Spores | 200 | Covers ~20% of byte offsets, enough for evolutionary selection |
| Oracle | Parses price from Borsh | Teacher/student asymmetry — Spores don't cheat |
| Pool address | Configurable CLI arg | Can target any Raydium/Orca pool |

---

## Open Questions (To Be Resolved During Implementation)

1. **Raydium v4 AmmInfo Borsh layout:** Exact byte offsets for `pool_coin_amount` and `pool_pc_amount`. Need to verify from Raydium SDK source or by inspecting a live account.
2. **Lookahead window:** How many records ahead for the Oracle? 100 records = ~100 slot updates. Need to calibrate based on actual data frequency.
3. **Helius API key management:** Environment variable vs. config file.
4. **Data volume:** 1 hour of SOL/USDC updates at ~1 update/second = ~3600 records = ~3.7 MB. Manageable.
