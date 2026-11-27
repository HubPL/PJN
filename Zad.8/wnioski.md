# Zadanie 8 – Kwantyzacja (bigscience/bloom-560m)

Prompt: "The most expensive cheese in the world is", generowanie greedy, max_length=100, czas to średnia z 3 przebiegów (GPU).

| Model / konfiguracja                                                | Rozmiar modelu [MB] | Czas generowania 100 tokenów [s] | Jakość tekstu |
| ------------------------------------------------------------------- | ------------------: | -------------------------------: | ------------- |
| BitsAndBytesConfig - NF4, double quant=True, compute_dtype=float16  | 634.6  | 3.417 | spójny, "very rare breed of sheep called Black" |
| BitsAndBytesConfig - NF4, double quant=False, compute_dtype=float16 | 634.6  | 3.232 | spójny, ten sam początek, potem inna kontynuacja |
| BitsAndBytesConfig - NF4, double quant=True, compute_dtype=float32  | 634.6  | 3.662 | spójny, prawie identyczny z NF4 + double quant + float16 |
| BitsAndBytesConfig - NF4, double quant=False, compute_dtype=float32 | 634.6  | 3.667 | spójny, zbliżony do NF4 bez double quant + float16 |
| BitsAndBytesConfig - FP4, double quant=True, compute_dtype=float16  | 634.6  | 3.244 | mniej spójny, zmyślone nazwy |
| BitsAndBytesConfig - FP4, double quant=False, compute_dtype=float16 | 634.6  | 3.206 | średnio spójny |
| BitsAndBytesConfig - FP4, double quant=True, compute_dtype=float32  | 634.6  | 3.917 | najsłabszy, pod koniec bełkot po włosku |
| BitsAndBytesConfig - FP4, double quant=False, compute_dtype=float32 | 634.6  | 3.738 | średnio spójny |
| load_in_8bit - 8-bit                                                | 778.6  | 7.261 | spójny |
| Pełna precyzja float32                                              | 2133.2 | 2.462 | spójny, inna treść niż w wersjach skwantyzowanych |

## Wnioski

- Kwantyzacja 4-bit zmniejsza model około 3,4 raza (2133 → 635 MB), a 8-bit około 2,7 raza (→ 779 MB).
- Rozmiar nie spada do 1/8, bo kwantyzowane są tylko warstwy Linear. Embeddingi Blooma (słownik ok. 250 tys. tokenów) zostają w fp16 i zajmują większość pamięci.
- Double quant nie zmienia wyniku `get_memory_footprint()`. Oszczędza tylko na stałych kwantyzacji, czyli kilkanaście MB, co przy tak małym modelu jest niewidoczne.
- Najszybszy jest model w pełnej precyzji. Przy małym modelu dekwantyzacja wag w locie spowalnia generowanie zamiast je przyspieszać.
- 8-bit jest najwolniejszy (ok. 3 razy wolniej niż float32) przez obsługę wartości odstających w LLM.int8.
- compute_dtype float32 jest wolniejszy od float16 o ok. 0,4–0,5 s. Double quant dokłada zwykle ok. 0,1–0,2 s.
- Każda kwantyzacja zmienia wygenerowany tekst względem float32 już od pierwszych tokenów (przy dekodowaniu greedy mała zmiana logitów zmienia cały tekst).
- NF4 daje spójniejsze teksty niż FP4. Na tekst bardziej wpływa double quant niż compute_dtype.
- Treść jest merytorycznie błędna we wszystkich wariantach, także w float32, bo bloom-560m to mały model.
- Przy tak małym modelu kwantyzacja opłaca się tylko ze względu na pamięć, nie na szybkość. Najlepszym kompromisem jest NF4 z compute_dtype float16.
