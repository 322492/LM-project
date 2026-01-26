=== BASELINE RUN SUMMARY ===
date: 2026-01-26T11:25:06
hyps: outputs\baseline\full_test.hyp.pl
refs: data\splits_random\test.pl
lines_used: 9124

model: facebook/nllb-200-distilled-600M
batch_size: 4
num_beams: 1
max_new_tokens: 96
seed: 123

BLEU: BLEU = 10.62 44.6/15.6/6.7/3.2 (BP = 0.963 ratio = 0.963 hyp_len = 207545 ref_len = 215416)
chrF: chrF2 = 37.69
