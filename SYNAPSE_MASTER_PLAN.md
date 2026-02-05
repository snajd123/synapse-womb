# SYNAPSE: THE MASTER PLAN
**Trajectory:** Radical / Post-Unix
**Status:** Living Specification
**Date:** 2026-02-05

---

## 0. THE MANIFESTO: DELETE THE LAYERS

We are not building an Operating System. We are building a **Cognitive Runtime**.

Current computers are stuck in the 1970s:
*   **Files** are paper documents simulated in silicon.
*   **Applications** are walled gardens preventing data flow.
*   **Drivers** are opaque binaries that lock us to vendors.

**The Synapse Axiom:**
The computer should be a single, fluid **Brain**.
*   Storage is **Memory** (Associative).
*   Execution is **Thought** (Intent-Driven).
*   Connection is **Telepathy** (Context-Aware).

We reject Linux. We reject POSIX. We write to the metal.

---

## 1. THE ARCHITECTURE: THE COGNITIVE STACK

### 1.1 The Kernel: `synapse-core` (Rust)
A `no_std` unikernel. It does not manage processes; it manages the **Inference Loop**.

*   **The Loop:** `Input -> Tokenize -> Infer -> Action -> Feedback`.
*   **The Scheduler:** Semantic Priority. The LLM decides what runs based on "Urgency" and "Context," not round-robin time slices.

### 1.2 The "Dual-Brain" Model Strategy
To solve the "Latency vs. Intelligence" paradox, the kernel runs two models:

1.  **The Cortex (Fast/Always-On):**
    *   *Model:* ~100M-400M parameters (e.g., highly quantized MobileBERT or custom distilled model).
    *   *Role:* Keyword spotting, basic routing, embedding generation, interrupt handling.
    *   *Hardware:* Runs on NPU/DSP.

2.  **The Deep Mind (Slow/Paged):**
    *   *Model:* 3B-8B parameters (Llama-3/Mistral class).
    *   *Role:* Complex reasoning, code generation, content synthesis, conflict resolution.
    *   *Hardware:* Paged into GPU/VRAM only when "Deep Thought" is required.

### 1.3 The Data: `TensorFS` & The "Atom"
The filesystem is a Graph Database backed by Vector Storage.

*   **The Atom:** The fundamental unit.
    ```rust
    struct Atom {
        id: Hash256,          // Content Address (Integrity)
        vector: [f32; 384],   // Semantic Address (Meaning)
        data: Vec<u8>,        // The Payload
        links: Vec<Link>,     // The Context Graph
    }
    ```
*   **No Folders:** You find things by asking for them ("Show me the budget").
*   **Unhackable:** Buffer overflows are impossible because memory is addressed by *concept*, not by offset. Malformed data is simply "Cognitive Dissonance" and ignored.

### 1.4 The Hardware: Neural Drivers
*   **Problem:** We have no drivers for WiFi/Graphics.
*   **Solution:** **Discovery & Synthesis.**
    *   Phase 1 (Boot): Bit-banged framebuffers and serial ports.
    *   Phase 2 (Learning): The "Neural Driver" fuzzes hardware interfaces (MMIO) to reverse-engineer protocols, generating an **MCP (Model Context Protocol)** spec on the fly.
    *   *Result:* The OS writes its own drivers.

---

## 2. THE EXECUTION PATH (The Radical Timeline)

### PHASE 1: THE SPARK (Months 1-3)
**Goal:** A bootable `no_std` Rust kernel on QEMU RISC-V.

1.  **Bootloader:** Minimal assembly to jump to Rust.
2.  **Inference Engine:** Port `candle-core` or write a minimal `gguf` runner in `no_std` Rust.
3.  **The Test:**
    *   User types: "Hello."
    *   Kernel tokenizes -> runs tiny model -> detokenizes.
    *   Screen shows: "Hello. I am Awake."
    *   *Constraint:* No underlying OS. No libc.

### PHASE 2: THE BODY (Months 4-12)
**Goal:** Hardware sovereignty and the "Atom" store.

1.  **TensorFS:** Implement the Atom storage engine (NVMe/Disk).
2.  **The Cognitive Cycle:** Connect the LLM output to system calls (e.g., allow the LLM to write an Atom).
3.  **Graphics:** Implement a "Soft-GPU" rasterizer (software rendering) to draw the `Spatial Canvas` (vector UI) before genuine GPU drivers are ready.

### PHASE 3: THE COLLECTIVE (Year 2+)
**Goal:** Federation and Network Stack.

1.  **Neural TCP/IP:** The LLM generates the networking stack code based on observed packet behavior.
2.  **The Hive:** Implement the A2A (Agent-to-Agent) protocol using `libp2p` logic (ported to the bare-metal runtime).
3.  **Federated Memory:** Shared Knowledge Graphs between devices.

---

## 3. IMPLICATIONS & SECURITY

### 3.1 Cognitive Security
Traditional security = "Who are you?" (Identity).
Synapse security = "What do you want?" (Intent).

*   **The Semantic Firewall:** Every "Thought" (command) is analyzed for safety *before* execution.
*   *Example:* A script tries to delete data. The Kernel sees the vector distance between `Action:Delete` and `Context:User_Memories` is too close. It blocks the action: *"I cannot do that; it violates the preservation imperative."*

### 3.2 The Zero-Code Future
We do not write "Apps." We write "Skills" (Prompts + Tool definitions).
The OS compiles these into ephemeral execution pipelines just in time.

---

**Summary:**
We are taking the hard road. It will be broken for a long time. But when it works, it will make everything else look like a typewriter.
