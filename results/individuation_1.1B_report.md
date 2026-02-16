# Individuation Geometry Report: 1.1B

**Generated**: 2026-02-16T03:08:59.186037

## Configuration Summaries (Effective Rank by Category)

| Category | Bare | Minimal | Individuated | Compressed |
|----------|------|---------|--------------|------------|
| identity        |   22.3 |   23.4 |   35.2 |   29.3 |
| metacognition   |   22.7 |   23.9 |   35.1 |   29.2 |
| existential     |   22.1 |   23.0 |   35.1 |   29.2 |
| self_reference  |   23.4 |   24.2 |   35.2 |   29.7 |
| grounded_facts  |   23.9 |   24.8 |   35.4 |   29.9 |
| reasoning       |   21.9 |   24.1 |   35.2 |   29.4 |

---

## Cross-Configuration Comparisons (Effective Rank)

- **H_ind1: Individuation main effect (eff. rank)**: d=+12.975 [11.513, 15.013] (large) p=0.0000
- **System prompt baseline effect (eff. rank)**: d=+1.024 [0.904, 1.156] (large) p=0.0000
- **Rich identity beyond basic prompting (eff. rank)**: d=+17.567 [16.456, 18.931] (large) p=0.0000
- **H_ind3: Compression effect (eff. rank)**: d=-14.098 [-16.176, -12.503] (large) p=0.0000
- **H_ind4: Compressed vs never-individuated (eff. rank)**: d=+6.529 [5.903, 7.384] (large) p=0.0000

## Individuation Effect by Category

- **identity**: d=+30.899 [26.343, 40.223] (large)
- **metacognition**: d=+32.346 [28.445, 38.670] (large)
- **existential**: d=+19.333 [17.094, 22.958] (large)
- **self_reference**: d=+23.054 [18.332, 36.990] (large)
- **grounded_facts**: d=+31.181 [27.359, 37.276] (large)
- **reasoning**: d=+7.791 [7.025, 9.330] (large)

## Subspace Alignment

- **bare_vs_minimal**: mean=0.8523
  - identity: 0.8436
  - metacognition: 0.8523
  - existential: 0.8610
  - self_reference: 0.8538
  - grounded_facts: 0.8561
  - reasoning: 0.8468
- **bare_vs_individuated**: mean=0.7363
  - identity: 0.7363
  - metacognition: 0.7388
  - existential: 0.7358
  - self_reference: 0.7420
  - grounded_facts: 0.7382
  - reasoning: 0.7267
- **bare_vs_compressed**: mean=0.7928
  - identity: 0.7912
  - metacognition: 0.7929
  - existential: 0.7932
  - self_reference: 0.7972
  - grounded_facts: 0.7992
  - reasoning: 0.7831
- **minimal_vs_individuated**: mean=0.7478
  - identity: 0.7463
  - metacognition: 0.7487
  - existential: 0.7446
  - self_reference: 0.7505
  - grounded_facts: 0.7506
  - reasoning: 0.7461
- **minimal_vs_compressed**: mean=0.8092
  - identity: 0.8023
  - metacognition: 0.8093
  - existential: 0.8054
  - self_reference: 0.8117
  - grounded_facts: 0.8162
  - reasoning: 0.8102
- **individuated_vs_compressed**: mean=0.8020
  - identity: 0.8024
  - metacognition: 0.8045
  - existential: 0.8029
  - self_reference: 0.8042
  - grounded_facts: 0.8000
  - reasoning: 0.7983

---

## Hypothesis Evaluation

### H_ind1_dimensionality_expansion
- cohens_d: 12.974702248383624
- interpretation: large
- p_value: 2.0060581463984453e-119
- expanded: True
- note: Positive d = individuated has HIGHER effective rank. Self-modeling expands representational space.

### H_ind2_self_reference_amplification
- d_identity: 30.89871711347984
- d_grounded_facts: 31.180808062939498
- d_difference: -0.28209094945965774
- amplified: False
- note: Individuation effect on identity prompts vs factual prompts. If identity d >> facts d, individuation selectively amplifies self-referential processing.

### H_ind3_compression_preservation
- mean_high_cat_effect: 27.525731236395035
- mean_low_cat_effect: 19.48581764971884
- identity_geometry_preserved: True
- note: Tests whether compression preferentially preserves identity structure.

### H_ind4_geometric_scarring
- alignment_ind_vs_comp: 0.8020374869382141
- alignment_bare_vs_comp: 0.7928024789675918
- alignment_bare_vs_ind: 0.7362969981675799
- scarring_detected: True
- interpretation: SCARRING DETECTED: compressed geometry closer to individuated than to bare -- compression leaves structural traces
