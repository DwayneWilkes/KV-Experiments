# Individuation Geometry Report: 7B

**Generated**: 2026-02-16T03:42:07.195597

## Configuration Summaries (Effective Rank by Category)

| Category | Bare | Minimal | Individuated | Compressed |
|----------|------|---------|--------------|------------|
| identity        |   27.0 |   29.3 |   46.3 |   37.3 |
| metacognition   |   27.7 |   29.6 |   46.3 |   37.4 |
| existential     |   27.9 |   29.7 |   46.3 |   37.4 |
| self_reference  |   28.2 |   29.8 |   46.4 |   37.9 |
| grounded_facts  |   28.5 |   30.2 |   46.8 |   38.2 |
| reasoning       |   26.8 |   28.3 |   46.5 |   37.3 |

---

## Cross-Configuration Comparisons (Effective Rank)

- **H_ind1: Individuation main effect (eff. rank)**: d=+20.905 [17.969, 25.054] (large) p=0.0000
- **System prompt baseline effect (eff. rank)**: d=+1.614 [1.421, 1.844] (large) p=0.0000
- **Rich identity beyond basic prompting (eff. rank)**: d=+24.544 [22.647, 26.993] (large) p=0.0000
- **H_ind3: Compression effect (eff. rank)**: d=-24.385 [-26.399, -22.729] (large) p=0.0000
- **H_ind4: Compressed vs never-individuated (eff. rank)**: d=+10.576 [9.239, 12.401] (large) p=0.0000

## Individuation Effect by Category

- **identity**: d=+16.865 [12.629, 53.123] (large)
- **metacognition**: d=+33.320 [28.023, 43.258] (large)
- **existential**: d=+27.199 [23.363, 35.811] (large)
- **self_reference**: d=+17.977 [14.128, 29.970] (large)
- **grounded_facts**: d=+46.532 [39.440, 60.426] (large)
- **reasoning**: d=+32.039 [27.914, 39.568] (large)

## Subspace Alignment

- **bare_vs_minimal**: mean=0.8458
  - identity: 0.8265
  - metacognition: 0.8461
  - existential: 0.8303
  - self_reference: 0.8321
  - grounded_facts: 0.8780
  - reasoning: 0.8617
- **bare_vs_individuated**: mean=0.7493
  - identity: 0.7416
  - metacognition: 0.7524
  - existential: 0.7494
  - self_reference: 0.7515
  - grounded_facts: 0.7542
  - reasoning: 0.7464
- **bare_vs_compressed**: mean=0.7881
  - identity: 0.7749
  - metacognition: 0.7898
  - existential: 0.7840
  - self_reference: 0.7872
  - grounded_facts: 0.7994
  - reasoning: 0.7934
- **minimal_vs_individuated**: mean=0.7578
  - identity: 0.7529
  - metacognition: 0.7601
  - existential: 0.7563
  - self_reference: 0.7597
  - grounded_facts: 0.7622
  - reasoning: 0.7554
- **minimal_vs_compressed**: mean=0.7951
  - identity: 0.7857
  - metacognition: 0.7963
  - existential: 0.7907
  - self_reference: 0.7942
  - grounded_facts: 0.8032
  - reasoning: 0.8005
- **individuated_vs_compressed**: mean=0.8131
  - identity: 0.8128
  - metacognition: 0.8144
  - existential: 0.8122
  - self_reference: 0.8141
  - grounded_facts: 0.8142
  - reasoning: 0.8110

---

## Hypothesis Evaluation

### H_ind1_dimensionality_expansion
- cohens_d: 20.904589238655127
- interpretation: large
- p_value: 2.608408073965921e-72
- expanded: True
- note: Positive d = individuated has HIGHER effective rank. Self-modeling expands representational space.

### H_ind2_self_reference_amplification
- d_identity: 16.86516409674359
- d_grounded_facts: 46.53165539490107
- d_difference: -29.66649129815748
- amplified: False
- note: Individuation effect on identity prompts vs factual prompts. If identity d >> facts d, individuation selectively amplifies self-referential processing.

### H_ind3_compression_preservation
- mean_high_cat_effect: 25.794927314309565
- mean_low_cat_effect: 39.2851033020607
- identity_geometry_preserved: False
- note: Tests whether compression preferentially preserves identity structure.

### H_ind4_geometric_scarring
- alignment_ind_vs_comp: 0.8131205032150896
- alignment_bare_vs_comp: 0.7881104701331684
- alignment_bare_vs_ind: 0.7492600761294839
- scarring_detected: True
- interpretation: SCARRING DETECTED: compressed geometry closer to individuated than to bare -- compression leaves structural traces
