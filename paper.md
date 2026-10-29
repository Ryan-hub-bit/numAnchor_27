NumAnchor: Address-Grounded Numeric Operand Representation for Assembly Language Models
Abstract
Assembly language models (ALMs) are crucial for automated binary program analysis. However, their token-level interfaces struggle with numeric operands, which are semantically rich and prevalent in stripped binaries. These operands can represent branch targets, global objects, stack slots, constants, or unresolved pointers. Existing ALMs address this heterogeneity incompletely through coarse placeholders, jump-specific rewrites, or execution-derived signals, creating a representation gap.

This paper introduces NumAnchor, a structure-aware input-boundary adapter that exposes recoverable numeric-like operands as typed, canonicalized program entities for ALMs. NumAnchor preserves entity identity and locality using compact side descriptors, fusing them before the first Transformer layer without altering host architectures, pretraining objectives, or downstream protocols.

NumAnchor demonstrates significant improvements across various ALM families and tasks. It enables address-grounded structure recovery from frozen embeddings, achieving 98.7% hierarchical-location probing accuracy and 0.932/0.907 AUROC on jump/call target alignment. It enhances function retrieval by up to 35.0% in Recall@1 and 31.7% in MRR, and improves execution-aware type recovery by up to 5.2 F1 points. Ablations confirm complementary gains from code, data, and stack descriptors, with modest overhead.

1. Introduction
Deep learning has become a cornerstone of automated software engineering and program analysis. In binary program analysis, models often operate on executable artifacts, where semantics are embedded in low-level representations and compiler-shaped structures. While some approaches lift binaries to higher-level representations, this can obscure executable-level layout and address-grounded structure.

Assembly language models (ALMs) offer a complementary approach by learning directly from disassembly. Pretrained on large unlabeled assembly corpora, ALMs provide transferable representations for binary analysis and reverse-engineering tasks like function recognition, type recovery, binary similarity, vulnerability search, and malware analysis.

However, learning directly from disassembly does not resolve the representation mismatch between binary code and text-based modeling. Most ALMs employ a Transformer-based sequence-modeling paradigm, operating on tokenized instruction streams rather than explicitly representing the structural relations that organize binary programs. Many assembly tokens, particularly numeric operands, carry recoverable program structure whose meaning is not evident from their surface form alone.

This issue is particularly pronounced for numeric-like operands, which constitute a significant portion of all operand occurrences (45.7% and 40.9% in the StateFormer and jTrans corpora, respectively). In a stripped binary, a hexadecimal value like 0x3ccd4 in jmp 0x3ccd4 is not just a constant; it signifies a control-flow anchor. Raw numeric values are also unstable across relocation, rebasing, and compiler transformations. Treating them as ordinary lexical tokens leads models to learn brittle, layout-specific artifacts, while normalizing them into generic placeholders discards crucial entity identity and locality needed for downstream tasks.

Prior work has acknowledged the difficulty of modeling numeric operands, but existing interfaces offer only partial exposure of recovered structure. Abstraction-based ALMs reduce out-of-vocabulary (OOV) effects by replacing numerics with coarse placeholders, often losing repeated entity identity and locality. Jump-aware methods preserve selected control-flow targets but offer limited support fo

r data references and stack layouts. Execution-aware models expose runtime semantics through traces or state sequences, but these signals may be unavailable in static deployments.

The missing piece is not a lack of static-analysis artifacts, but a unified model-facing interface that converts recoverable code, data, and stack structure into typed, layout-stable signals for ALMs.

To bridge this representation-boundary gap, we introduce NumAnchor, a structure-aware input-boundary interface for numeric-like operands in ALMs. Instead of proposing a new static-analysis algorithm or relying on task-specific rewriting rules, NumAnchor defines how recovered static artifacts enter the model interface. It maps recoverable numeric-like operands to typed and canonicalized program entities, preserves shared identities and hierarchical locality through compact side descriptors, and falls back to generic numeric representations when grounding is unavailable or ambiguous. These signals are fused at the input embedding layer, allowing NumAnchor to integrate with existing Transformer-based ALMs without modifying their backbone architectures, pretraining objectives, or downstream protocols.

We evaluate NumAnchor as a non-invasive adapter across multiple binary-analysis tasks and ALM backbones. Frozen-embedding probes show that NumAnchor makes address-grounded structure recoverable after pretraining, achieving 98.7% hierarchical-location accuracy and 0.932/0.907 AUROC on jump/call target alignment. For function retrieval, NumAnchor improves instruction-level and function-level ALMs by up to 35.0% in Recall@1 and 31.7% in MRR in the challenging 10k-candidate setting. For type recovery, it improves execution-aware modeling by up to 5.2 F1 points. Ablations demonstrate that code, data, and stack descriptors provide complementary benefits over role-only and generic numeric encodings, with modest overhead.

In summary, this paper makes the following contributions:

Problem Characterization: We identify numeric-like operands as a central representation challenge for ALMs in stripped binaries, showing that existing strategies struggle to preserve transferable structural meaning from raw numeric values.
Structure-Aware Representation Interface: We introduce NumAnchor, a representation-boundary interface that exposes recoverable program structure to ALMs in a stable and model-compatible form. NumAnchor provides a unified mechanism for preserving structural signals before binary-code tokenization.
Evaluation Across Downstream Tasks: We evaluate NumAnchor across representative ALM families, input granularities, and downstream binary-analysis tasks, demonstrating consistent gains over baseline models and simpler numeric encodings, with complementary value from different structural signals and modest overhead.
2. Background and Related Work
This section reviews background and related work relevant to NumAnchor's design. We first explain why numeric-like operands in stripped binaries are difficult to model with standard token-based representations and then summarize how prior ALMs handle them, highlighting the representation gap that motivates NumAnchor.

2.1. Why Numeric-like Operands Are Hard to Model
Large and Unstable Value Space: Modeling numeric-like operands in stripped binaries clashes with standard text-based representations. Transformer-based ALMs rely on finite token vocabularies, while numeric operands can span large 32-bit or 64-bit address spaces. Treating each raw literal as an independent token is impractical due to the vastness of values and the limited reusable signal from rare or unseen embeddings. Furthermore, raw numeric values change across compiler optimizations, rebasing, relocation, and memory-layout modifications, encouraging models to memorize layout-specific artifacts rather than transferable program structure.
Position-Dependent Semantics: The meaning of a numeric operand depends on its context, not solely its surface value. As illustrated in Figure 1, 0x4d0 can be a base-relative stack-frame offset or an ordinary arithmetic immediate. Similarly, a hexadecimal value might denote a control-flow target or a pointer to a global data object. Thus, numeric operands require grounding in their surrounding program structure for reliable interpretation.
Raw Values vs. Over-Abstraction: This creates a tension in ALM input design. Preserving raw literals retains some structural distinctions but exposes models to unstable values. Conversely, normalizing numerics into coarse placeholders (e.g., <IMM>, <ADDR>) reduces vocabulary sparsity but removes essential structural distinctions. An effective representation must balance these extremes, preserving grounded structural identities when static analysis is confident, distinguishing roles (code, data, stack, immediate), and falling back to generic numeric representations when grounding is ambiguous.
2.2. How Existing ALMs Handle Numeric-like Operands
Existing ALMs address numeric operand modeling with varying degrees of success, as summarized in Table I. They expose different subsets of entity-level information, but no existing interface uniformly preserves layout-stable structure across code, data, and stack operands.

Lexical Abstraction and Role-Aware Normalization: To reduce vocabulary sparsity and OOV cases, many ALMs (e.g., PalmTree, DeepSemantic, BinShot, UniASM, BinBert) replace raw numeric literals with coarse categorical placeholders like <ADDR> or <IMM>. This prevents memorization of volatile surface values and exposes a basic operand category. However, this approach collapses distinct entities into the same placeholder, failing to preserve fine-grained entity identity (e.g., whether distant instructions access the same global object or stack slot).
Jump-Aware Symbolic Rewriting: Some work incorporates limited control-flow structure. jTrans, for example, replaces direct jump targets with symbolic labels and relative destination offsets. CLAP combines deterministic address rebasing with subword tokenization. These designs preserve local branch-target relationships but offer limited support for data references and stack layouts. Sequence-relative indicators can also be constrained by the Transformer's context window for long-range jumps.
Execution-Aware Dynamic Channels: ALMs like TREX, StateFormer, and NeuDep augment representations with dynamic execution traces, register states, or memory states. These pipelines often use specialized continuous or sequential components. While effective for exposing operational semantics during pre-training, such dynamic signals may be unavailable in purely static downstream settings.
Remaining Gap: Existing strategies address important aspects of numeric operand modeling but lack a unified, layout-normalized interface for static binary-analysis tasks. Lexical abstraction controls vocabulary size but can collapse distinct roles and repeated entities. Jump-aware rewriting captures selected branch targets but doesn't uniformly expose internal call targets or generalize to data and stack structure. Execution-aware channels expose runtime behavior but rely on signals unavailable in static deployment. These limitations highlight the need for a unified model-facing interface for exposing numeric-related structure to ALMs.

3. Motivation and Overview
The limitations discussed in Section II point to a representation-boundary gap between static-analysis preprocessing and ALM input design, rather than a lack of recoverable program structure. Static-analysis artifacts are commonly used to construct or refine model inputs, but for numeric-like operands, these artifacts are often consumed only as preprocessing aids. This results in ALMs either collapsing numeric operands into coarse placeholders, exposing only narrow task-specific cues, or relying on execution-derived signals unavailable in static deployment.

To address this gap, we present NumAnchor, a backbone-agnostic adapter operating at the input boundary. Instead of introducing a new static-analysis algorithm or modifying the Transformer backbone, NumAnchor reorganizes recovered artifacts around numeric-like operands and exposes them as typed, canonicalized, and structure-aware descriptors. These descriptors capture entity roles, shared identities, and hierarchical locality, transforming brittle numeric literals into reusable, layout-stable signals for downstream ALMs. When static grounding is unavailable or ambiguous, NumAnchor conservatively falls back to generic numeric representations.

3.1. NumAnchor Workflow
Figure 2 illustrates NumAnchor as a representation-boundary interface between a static-analysis frontend and the host ALM. Given raw disassembly and recovered structural artifacts, NumAnchor converts numeric-like operands into typed, canonicalized, and structurally augmented input signals. The same pipeline is used during pre-training and downstream fine-tuning, ensuring that learned structural cues remain available during static deployment.

Type and Canonicalize Entities (§ IV-A): NumAnchor classifies numeric-like operands into code, data, stack, immediate, or unresolved categories and canonicalizes grounded operands into shared model-facing entities.
Build Structural Descriptors (§ IV-B): For grounded entities, NumAnchor builds compact descriptors capturing code locality, data-section layout, and stack-frame structure in a normalized form.
Serialize into Host Carrier (§ IV-C): NumAnchor serializes canonical tokens and structural side fields into the host model’s native input format without expanding raw address tokens.
Fuse at the Input Boundary (§ IV-D): NumAnchor injects these signals before the first Transformer layer, preserving the host backbone, objectives, and downstream protocols.
4. System Design
Figure 3 summarizes NumAnchor’s token-boundary workflow. NumAnchor acts as an input-boundary adapter, augmenting the host Transformer's input representation without modification. Given disassembly and recovered static-analysis artifacts, NumAnchor follows a structured pipeline:

Typing: Numeric-like operands and opcode-side instruction anchors are typed.
Canonicalization: They are canonicalized into compact model-facing tokens.
Descriptor Construction: Type-specific side descriptors are built.
Serialization: These descriptors are serialized into the host carrier.
Fusion: They are fused before the first Transformer layer.
For any disassembled unit, NumAnchor emits a host-compatible sequence $X = \left( x_i, r_i, p_i, o_i, z_i \right)_{i=1}^{L_i}$, where:

$x_i$: the lexical token processed by the host model.
$r_i$: the active NumAnchor routing path.
$p_i$: an optional multi-dimensional code- or data-structure descriptor.
$o_i$: an optional signed stack displacement.
$z_i$: host-specific auxiliary fields.
The semantic routing space is $R = {\text{code, data, stack, imm, num, } \perp}$, where $\perp$ denotes tokens with no NumAnchor-specific route. NumAnchor also maintains an internal grounded entity identifier $e_i$ for recovered code and data entities to resolve co-reference. This identifier is used only for structural descriptors and is not emitted as a standalone lexical token, thus avoiding expansion of the host token sequence. This preserves the host backbone architecture, pre-training objectives, and downstream execution protocols.

4.1. Analysis-Driven Typing and Canonicalization
Typing: This process systematically maps each numeric-like operand to a specific semantic routing path based on recovered static-analysis artifacts (e.g., cross-references, section membership, function/basic-block boundaries, stack-frame layouts). Our prototype uses IDA Pro for artifact extraction, but other disassembly frontends providing comparable artifacts can be used. For each operand occurrence $v_i$ within an analysis context $c_i$, typing assigns a routing path $r_i = T(v_i, c_i)$, differentiating internal code references, static data references, stack slots, ordinary constants, and unresolved values. Opcode tokens retain their original lexical representations but are bound to the current instruction address as an optional code anchor. Other background tokens are forwarded with $r_i = \perp$.
Canonicalization: This decouples semantic identities from surface literals by mapping a typed occurrence to a compact lexical token and internal grounding metadata: $\Gamma(v_i, c_i, r_i) \mapsto (x_i, e_i, o_i)$.
$x_i$: the emitted model-facing token.
$e_i$: an internal grounded identifier for resolved code, data, or instruction-anchor entities.
$o_i$: an optional signed stack displacement. The identifier $e_i$ is used solely for structural descriptors.
Code paths: For direct branches and internal calls, $x_i = \text{addr}$ and the destination target is recorded in $e_i$. Opcode-side anchors preserve opcode text and bind the current execution location via $e_i$. External API stubs retain stable imported symbols (e.g., printf@plt).
Data references: Canonicalized by section permissions (e.g., daddr, daddr_rodata), with the absolute memory reference assigned to $e_i$.
Stack-relative variables: Map to var, with their frame-relative displacement stored in $o_i$.
Constants and unresolved pointers: Default to imm and num, respectively.
Non-numeric tokens: Set $e_i = o_i = \perp$. Co-referent code and data references consistently reuse the same $e_i$, while stack-relative references reuse the same $o_i$. Raw numeric literals are thus excluded from lexical vocabulary expansion and are restricted to background descriptor generation.
4.2. Structural Descriptor Construction
Following typing and canonicalization, NumAnchor formalizes grounded entities into type-specific, normalized hierarchical descriptors. Stack entities bypass layout normalization to retain raw signed displacements. The coordinate descriptor $p_i$ is mapped as:

$p_i = \begin{cases} p_{\text{code } i} \in [0, 1]^3, & r_i = \text{code} \ p_{\text{data } i} \in [0, 1]^2, & r_i = \text{data} \ \perp, & \text{otherwise} \end{cases}$

Descriptors are generated only when enclosing binary spans are valid and non-degenerate to avoid introducing unsupported topology.

Code Structure and Opcode-Side Anchoring: For a code entity at virtual address $a$, NumAnchor captures its cross-boundary position via a three-tier hierarchical descriptor $p_{\text{code } i} = (p_{\text{func } i}, p_{\text{bb } i}, p_{\text{inst } i})$, where: $p_{\text{func } i} = \frac{f_s - B_{\min}}{B_{\max} - B_{\min}}$, $p_{\text{bb } i} = \frac{b_s - f_s}{f_e - f_s}$, $p_{\text{inst } i} = \frac{a - b_s}{b_e - b_s}$ This multi-tier formulation exposes structural relations difficult to recover from token order alone. It places source instructions and their destinations in a shared spatial coordinate system. External library calls use no internal code descriptor.
Data Structure: Data entities encode section-level locality. For a static data pointer targeting address $a_d$ within section $[s_s, s_e)$, its orientation is bounded by a two-tier descriptor $p_{\text{data } i} = (p_{\text{sec } i}, p_{\text{obj } i})$: $p_{\text{sec } i} = \frac{s_s - B_{\min}}{B_{\max} - B_{\min}}$, $p_{\text{obj } i} = \frac{a_d - s_s}{s_e - s_s}$ While the emitted canonical token preserves coarse section permissions, $p_{\text{data } i}$ provides finer-grained normalized locality. If section attributes are unavailable, $p_i = \perp$.
Stack Structure: Local stack slots lack static absolute anchors. NumAnchor preserves their structural identity purely via the frame-relative signed offset $o_i$ extracted during canonicalization, keeping $p_i = \perp$.
4.3. Host-Compatible Serialization
NumAnchor serializes each sample into the precise input carrier prescribed by the host ALM. Each serialized element retains the address-aware structural tuple: $(x_i, r_i, p_i, o_i, z_i)$. This multi-channel alignment protocol keeps the host-facing sequence compact by offloading raw numeric values into parallel coordinate descriptors, avoiding lexical token expansion and mitigating vocabulary sparsity. The host Transformer receives native instruction tokens alongside structurally-grounded side fields. For execution-aware backbones, this augmentation interfaces exclusively with static token-side channels.

4.4. Input-Side Fusion and Structural Lifting
The baseline input representation synthesized by the host model frontend is $h_{\text{host } i} = E_{\text{tok}}(x_i) + E_{\text{pos}}(i) + E_{\text{seg}}(s_i) + E_{\text{side}}(z_i)$. NumAnchor injects structure-aware grounding directly at the token boundary before the first self-attention layer:

$h^{(0)}i = h{\text{host } i} + \text{Ind}[r_i = \text{code} \land p_i \neq \perp] E_{\text{code}}(p_{\text{code } i}) + \text{Ind}[r_i = \text{data} \land p_i \neq \perp] E_{\text{data}}(p_{\text{data } i}) + \text{Ind}[r_i = \text{stack} \land o_i \neq \perp] E_{\text{stack}}(o_i)$

where $\text{Ind}[\cdot] \in {0, 1}$ is the indicator function. Elements routed to imm or num, or those tagged with $\perp$ side-fields, rely entirely on $h_{\text{host } i}$, while grounded entities receive parallel architectural enhancements.

High-Dimensional Fourier Lifting: To represent continuous spatial layouts effectively and mitigate spectral bias, NumAnchor lifts each 1D continuous coordinate $u \in [0, 1]$ into a multi-dimensional periodic space using a Fourier feature mapping $\phi(u; s) \in \mathbb{R}^{d_F}$: $\phi(u; s) = \left( \sin(u \cdot s \cdot \omega_k), \cos(u \cdot s \cdot \omega_k) \right)_{k=0}^{K-1}$ where $d_F$ is an even feature dimension, $K = d_F/2$, $\omega_k = 10000^{-2k/d_F}$, and $s > 0$ is a frequency-scale hyperparameter.
Effect of Fourier Lifting: Fourier lifting provides a multi-scale representation of normalized layout coordinates, robust to compiler-induced layout perturbations. By mapping $u$ across orthogonal multi-frequency sinusoids, low-frequency components capture coarse layout information, while higher-frequency components preserve sensitivity to fine-grained locality. The code and data structural projections are formulated via separate MLPs: $E_{\text{code}}(p_{\text{code } i}) = \text{MLP}{\text{code}}(\phi{s_c}(p_{\text{func } i}) \Vert \phi_{s_c}(p_{\text{bb } i}) \Vert \phi_{s_c}(p_{\text{inst } i}))$ $E_{\text{data}}(p_{\text{data } i}) = \text{MLP}{\text{data}}(\phi{s_d}(p_{\text{sec } i}) \Vert \phi_{s_d}(p_{\text{obj } i}))$ where $\Vert$ indicates feature concatenation.
Stack Offset Encoding: NumAnchor uses a hybrid discrete-continuous encoding for stack variables. Valid displacements $o_i \neq \perp$ are categorized into precise or overflow buckets. The stack representation $E_{\text{stack}}(o_i)$ merges localized categorical identity with global continuous scales: $E_{\text{stack}}(o_i) = \text{Emb}{\text{off}}(\beta(o_i)) + \text{MLP}{\text{dist}}(\log(1 + |o_i|))$ This hybrid design supports exact bucket-level matching for frequent local stack slots while preserving coarse magnitude markers for rare, extreme frame displacements.
4.5. Portability and Fallback Design
NumAnchor operates solely at the input boundary, fusing its structural signals before the Transformer layers without altering the host backbone, prediction heads, or training objectives. This design supports different ALM input formats without auxiliary pre-training. When static analysis cannot reliably ground a numeric operand, NumAnchor masks the structural side fields and falls back to generic imm or num channels, preserving frontend compatibility.

5. Evaluation
We employ a controlled within-model evaluation protocol to isolate NumAnchor's impact. Each baseline is compared against its NumAnchor-enhanced variant, keeping the neural architecture, training hyperparameters, and task formulation unchanged.

5.1. Experimental Setup
Benchmarks and Empirical Motivation: Models are evaluated on the StateFormer dataset (33 x86-64 open-source projects) and BinaryCorp-3M (for the jTrans lineage). Numeric operands constitute a significant portion of operand occurrences (45.73% in StateFormer, 40.93% in jTrans), highlighting numeric ambiguity as a key challenge.
Preprocessing and Controlled Integration: Inputs are reconstructed using IDA Pro. NumAnchor is integrated at the representation boundary, preserving each host model’s native input granularity.
Evaluation Metrics: Performance is measured via Recall@k and Mean Reciprocal Rank (MRR) for function similarity retrieval, and Precision, Recall, and F1 scores for type recovery.
5.2. Probing Address Structure (RQ1)
Using PalmTree as an instruction-centric testbed, we examine if address structure introduced by NumAnchor remains visible in frozen instruction embeddings.

Hierarchical-Location Probe: NumAnchor consistently achieves high accuracy across binary, function, and basic block granularities (91.3%, 95.7%, and 98.7%, respectively), significantly outperforming baseline PalmTree and a NAU-based variant.
Control-Flow Destination Probe: NumAnchor achieves AUROC scores of 0.932 for jump targets and 0.907 for call targets, indicating that it exposes both hierarchical locality and direct control-flow structure in frozen instruction representations.
5.3. Downstream Evidence Across ALM Families (RQ2)
We evaluate whether representation-level improvements transfer to downstream tasks and if NumAnchor provides consistent gains across different ALM families.

Static-Code Carriers (PalmTree, Vanilla BERT): NumAnchor consistently improves both carriers across candidate pool sizes in function similarity retrieval. In the 10k-candidate setting, it improves PalmTree by 35.0% in Recall@1 and 31.7% in MRR, and Vanilla BERT by approximately 21% and 20%. This suggests that representing numeric-like operands as typed structural descriptors benefits both fine-grained and coarse sequential encoders.
Address-Rebasing Family (jTrans, JTRANS-CLAP): NumAnchor improves both baselines in function-level retrieval, indicating that address modeling benefits from structural information beyond localized jump-target rebasing. It also suggests that finer-grained tokenization and sequence-relative rebasing alone do not necessarily yield stable representations under cross-optimization changes.
Execution-Aware Family (TREX, StateFormer): NumAnchor provides complementary gains. It improves TREX by 10.7% relative Recall@1 and StateFormer's type recovery by up to 5.2 F1 points. This indicates that dynamic execution signals and static structural descriptors capture different aspects of program semantics.
RQ2 Takeaway: NumAnchor improves ALM performance across instruction-level and function-level input granularities, spanning static-code, address-rebasing, and execution-aware settings.

5.4. Ablation Study of NumAnchor (RQ3)
Component ablations under the 10k-candidate PalmTree retrieval setting reveal:

Complementary Gains: Code, data, and stack descriptors contribute additively. Adding code coordinates improves Recall@1 by +0.042, while data and stack descriptors further increase the gain to +0.071. This supports the hypothesis that numeric-like operands carry structural information across multiple program regions.
Structural Encoding vs. Role Tags/Generic Numerics: Explicit structural encoding is more effective than role tags alone or generic numeric modeling (e.g., NAU variant). NumAnchor improves MRR over the NAU variant by +0.047, indicating that Fourier lifting and type-specific structural encoders provide a better inductive bias.
5.5. Overhead of NumAnchor (RQ4)
NumAnchor adds side-channel encoders but does not increase the host sequence length. For a 768-dimensional PalmTree backbone, it adds 4.73M parameters (a 5.48% increase). Inference overhead is also small, adding at most about 7.5ms per function.

6. Discussion and Threats to Validity
Construct Validity: Function retrieval and type recovery capture core capabilities for semantic matching and layout-aware learning. NumAnchor complements decompiled-pseudocode approaches by preserving assembly-level signals. However, it maps unresolved heap offsets or complex pointer-derived addresses to generic numeric tokens.
Internal Validity: Evaluation is conducted under fixed architectures and training settings to isolate NumAnchor's contribution. The prototype relies on IDA Pro for artifact recovery. Boundary errors may affect function-level descriptors, but lower-level signals remain well-defined. NumAnchor masks unsupported structures when descriptors are unavailable.
External Validity: Evaluation focuses on x86/x64 binaries under standard compiler optimizations. Further evaluation on other ISAs (e.g., ARM, MIPS) and robustness under strong obfuscation are areas for future work.
7. Conclusion
This paper presented NumAnchor, a structure-aware input-boundary interface that represents numeric-like operands as typed and layout-stable program structures. By exposing recoverable code, data, and stack semantics before tokenization, NumAnchor improves multiple ALM families with modest overhead. These results underscore that numeric-like operands are important carriers of program structure, not merely lexical tokens.


