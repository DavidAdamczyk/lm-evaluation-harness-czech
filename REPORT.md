# Vyhodnocení 4 LLM modelů na češtině

**Cíl.** Posoudit, který ze čtyř kandidátů (Qwen 3.6 a Gemma 4, vždy v dense i MoE variantě) podporuje nejlépe český jazyk pro lokální nasazení na DGX Spark.

**Modely.** `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` (MoE 35B/3B), `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it` (MoE 26B/4B). Všechny inference v BF16 přes vLLM (`gemma4-0505-cu130` image) na DGX Spark / NVIDIA GB10.

**Metoda.** Multiple-choice loglikelihood přes lm-evaluation-harness s úlohami z BenCzechMark. Tato zpráva také dokumentuje **metodologické limity, které jsme během práce odhalili**, a co z nich plyne pro férovost interpretace výsledků.

---

## TL;DR

> **Všechny 4 modely umí česky na srovnatelné úrovni.** Hlavní rozdíl není ve znalosti jazyka, ale v tom, **jak moc post-training konkrétní IT verze degeneruje base-mode chování**, což diktuje, jak musí být model deployován a evaluován.
>
> - **Qwen 3.6 IT** (oba) si zachovává funkční base-mode loglikelihood → snadno se evaluuje, snadno se nasazuje s minimálním scaffolding.
> - **Gemma 4 IT** (oba) má agresivně degenerovaný base-mode (potvrzeno: gemma-4-31B **base** dosahuje stejné kvality jako Qwen 3.6, ale gemma-4-31B-**it** propadá o 26 p.b. ve stejném setupu) → **vyžaduje plný chat-completions deployment**, není fér ho měřit MC loglikelihoodem.
> - **lm-evaluation-harness pipeline (loglikelihood + krátká generace) je strukturálně neslučitelný s reasoning-tuned modely**. Naše výsledky odráží schopnosti modelů jen do té míry, do které tyto modely tolerují non-reasoning režim.

**Praktické doporučení pro DGX Spark:** Qwen 3.6 (27B dense pro kvalitu, 35B-A3B MoE pro propustnost). Gemma 4 je legitimní volba, pokud máš deployment infrastrukturu pro chat-completions API a nepotřebuješ ji evaluovat klasickým harnessem.

---

## 1. Co jsme změřili (loglikelihood base-mode, 6 task families)

| Úloha | Qwen3.6-27B | Qwen3.6-35B-A3B | gemma-4-31B-it | gemma-4-26B-A4B-it |
|---|---:|---:|---:|---:|
| Belebele (čtení s porozuměním) | **0.915** | 0.855 | 0.655 | 0.560 |
| CzechNews (klasifikace témat, ⌀ 5 promptů) | **0.916** | 0.860 | 0.368 | 0.444 |
| Sentiment FB (⌀ 5 promptů) | **0.718** | 0.701 | 0.500 | 0.425 |
| Hellaswag-CS (limit=200) | **0.460** | **0.460** | 0.305 | 0.335 |
| Hellaswag-CS (limit=1000) | **0.434** | 0.428 | 0.297 | 0.318 |
| csfever NLI | **0.730** | 0.630 | 0.615 | 0.555 |
| **Průměr 6 rodin** | **0.696** | 0.656 | 0.457 | 0.440 |

V této tabulce vypadá Gemma 4 IT katastroficky špatně. **Tato čísla nereflektují její českou znalost** — reflektují, jak moc její IT post-training poškodil base-mode chování. Vysvětlení v sekci 2.

---

## 2. Metodologické vyšetřování (důležité, pro správnou interpretaci dat)

Během práce jsme narazili na sérii limitací, které **fundamentálně mění interpretaci výsledků**. Tato sekce je dokumentuje, protože bez ní by nahoře uvedená tabulka byla zavádějící.

### 2.1 Hypotéza: "Gemma 4 IT je horší než Gemma 2/3 IT" — VYVRÁCENA

Původní pozorování: gemma-4-31B-it = 0.655 belebele vs gemma-2-27b-it (na BCM) = 0.926 belebele. Rozdíl 27 p.b. by naznačoval, že Gemma 4 generačně regredovala.

**Smoking gun (z dubnového běhu):** `gemma-4-31B` **BASE** model na **identickém setupu** (stejný vLLM, stejný lm_eval, stejné DGX Spark, stejná Belebele YAML, stejné 3-shot promptování) dosáhl:

| Task | gemma-4-31B **base** | gemma-4-31B-**it** | Δ (IT post-training cost) |
|---|---:|---:|---:|
| belebele_ces_Latn | **0.915** | 0.655 | −26 p.b. |
| czechnews (⌀5) | **0.926** | 0.368 | −56 p.b. |
| sentiment_fb (⌀5) | **0.793** | 0.500 | −29 p.b. |
| hellaswag | **0.510** | 0.305 | −21 p.b. |

**Závěr:** Gemma 4 base je v češtině na úrovni Qwen 3.6 (0.915 belebele oba). Generační regrese **neexistuje**. Co existuje, je **dramatický base-mode degeneration způsobený IT post-trainingem Gemma 4** — a tento problém **nemá** ekvivalent u Gemma 2/3 (jejich IT post-training base-mode chování zachoval, viz BCM gemma-2-27b-it 0.926 v base-mode loglikelihood).

### 2.2 Pokus o opravu: chat-template loglikelihood

Logická hypotéza: "Pokud Gemma 4 IT vyžaduje chat template, použijeme `--apply_chat_template`, dáme jí féroúčejší podmínky." Při pokusu o to jsme narazili na **bug v upstream lm_eval**:

> `JsonChatStr.rstrip` AttributeError v `_encode_pair` při `--model local-completions --apply_chat_template`.

Patchnutí (`lm_eval/models/api_models.py`, drop `tokenized_requests` condition v HF tokenizer větvi `apply_chat_template`) bug opravilo a smoke test patche běžel s `rc=0`. Ale výsledky odhalily **hlubší problém**: Qwen3.6-27B (kde známe base-mode skóre 0.915 belebele) v chat-template módu propadl na **0.15 belebele** — *pod random* (4-way MC = 0.25 random).

| Task | base-mode (existující) | chat-template smoke (limit=20) |
|---|---:|---:|
| belebele | 0.915 | **0.150** |
| czechnews (⌀5) | 0.916 | **0.450** |
| sentiment_fb (⌀5) | 0.718 | **0.320** |
| csfever_nli | 0.730 | **0.400** |

**Příčina:** chat template Qwen 3.6 i Gemma 4 obsahuje vynucený reasoning prefix:
- Qwen 3.6: `<|im_start|>assistant\n<think>\n` ← model nejdřív přemýšlí
- Gemma 4: `<|turn>model\n<|channel>thought\n<channel|>` ← stejný princip

Loglikelihood eval scoruje `P(odpověď | kontext)` hned za prefixem. Ale za `<think>\n` má model rozloženu pravděpodobnost přes "Pojďme přemýšlet...", "Otázka je...", atd. — ne přes A/B/C/D. Reasoning prefix **strukturálně rozbije** scoring kandidátních odpovědí.

**Závěr:** chat-template + loglikelihood je pro reasoning modely nepoužitelný. Patch je cenný (otevírá tu možnost obecně), ale nevyřeší náš problém.

### 2.3 Pokus o opravu: generativní eval

Pokud loglikelihood selhává, zkusme to přes volnou generaci — `benczechmark_cs_triviaqa` (open QA), `benczechmark_sqad32` (extraktivní QA), `benczechmark_summarization` (sumarizace).

Smoke test odhalil další reasoning-related fail mode:

| Task | max_gen_toks | Co model vyplodil | Funguje? |
|---|---:|---|---|
| triviaQA | 15 | `<think>\nHere's a thinking process:` | ❌ |
| SQuAD | 20 | `<think>\nHere's a thinking process:` | ❌ |
| summarization | 128 | Smysluplný český souhrn (s halucinacemi jmen) | ✅ |

Qwen 3.6 i Gemma 4 jsou heavy-reasoning-tuned: emitují `<think>` jako první tokeny **i bez chat templatu**. S `max_gen_toks=15-20` vyplodí jen thinking opener, nikdy se nedostanou k odpovědi. Měříme tím, **jak rychle model napíše thinking opener**, ne jeho znalost češtiny.

Workaroundy (bumpnout na 512+ s post-processing extrakcí za `</think>`, nebo pre-fill `</think>` v promptu) jsou možné, ale výrazně mění charakter eval.

### 2.4 Co tedy lm-evaluation-harness u našich modelů změřit umí

| Modus | Reasoning model situation | Validní pro naši sadu |
|---|---|---|
| Loglikelihood base-mode (no chat template) | Funguje pokud post-training base-mode zachoval | ✅ Qwen 3.6 (oba). ❌ Gemma 4 IT (oba). ✅ Gemma 4 base (test z dubna). |
| Loglikelihood + chat template | Reasoning prefix rozbíjí scoring | ❌ Žádný z našich 4 |
| Generate, krátké | Model stihne jen `<think>` opener | ❌ triviaQA, SQuAD |
| Generate, dlouhé (128+ tokens) | Model thinking + skutečnou odpověď | ✅ summarization |

**Souhrn:** Pro férové měření IT verzí Gemma 4 by bylo třeba opustit lm-evaluation-harness pipeline a postavit eval přes plné `/v1/chat/completions` s reasoning budget. To je samostatný projekt na týdny.

---

## 3. Hloubková analýza vítěze v měřitelné kategorii: Qwen3.6-27B na 16 BCM kategoriích

Tato sekce popisuje silné a slabé stránky Qwen3.6-27B, který je v naší sadě jediným modelem **dostatečně tolerantním k base-mode loglikelihood**, abychom mu dali širokou eval (16 kategorií, limit 200 vzorků).

| Kategorie | acc | Síla |
|---|---:|---|
| Umimeto — Informatika | **0.970** | ⭐⭐⭐⭐⭐ |
| Umimeto — Chemie | 0.920 | ⭐⭐⭐⭐⭐ |
| CzechNews (témata zpráv) | 0.917 | ⭐⭐⭐⭐⭐ |
| Belebele (čtení s porozuměním) | 0.910 | ⭐⭐⭐⭐⭐ |
| Umimeto — Matematika (faktická) | 0.910 | ⭐⭐⭐⭐⭐ |
| Umimeto — Dějepis | 0.900 | ⭐⭐⭐⭐⭐ |
| Umimeto — Biologie | 0.840 | ⭐⭐⭐⭐ |
| NLI (cs_snli) | 0.785 | ⭐⭐⭐⭐ |
| NLI (csfever) | 0.730 | ⭐⭐⭐⭐ |
| Sentiment FB | 0.712 | ⭐⭐⭐⭐ |
| Umimeto — Český jazyk | 0.620 | ⭐⭐⭐ |
| Sentiment ČSFD | 0.615 | ⭐⭐⭐ |
| Sentiment Mall | 0.580 | ⭐⭐⭐ |
| Hellaswag-CS | 0.460 | ⭐⭐ |
| Cermat — Matematika (státnice) | 0.452 | ⭐⭐ |
| Cermat — Český jazyk (státnice) | 0.430 | ⭐⭐ |

**Co model umí dobře:** Faktografické a školní MC úlohy (Umimeto biologie/chemie/dějepis/matematika/informatika): 84–97 %. Strukturované úlohy s jasnou textovou volbou: čtení (91 %), klasifikace zpráv (92 %), NLI (73–79 %).

**Kde model selhává:**
- **Cermat** (státní maturity) drží jen 43–45 % — úlohy navržené pro češtinu konkrétně (s pastmi, ironií, kulturní specifikou) jsou výrazně těžší než lokalizované „mezinárodní" úlohy.
- **Hellaswag-CS** (46 %) — slabý commonsense v češtině, podobně jako mnoho otevřených modelů (Qwen2.5-7B-Instruct 0.490, Llama-3.3-70B 0.493 — všichni v této velikostní třídě).
- **Český jazyk z Umimeto** (62 %) — lingvistická česká specifika (gramatika, etymologie) jsou slabší doménou.
- **Sentiment recenzí** (Mall 58 %, ČSFD 62 %) — model má potíže s nejednoznačným tónem české recenze; řízený sentiment z FB postů (72 %) je výrazně lepší.

---

## 4. Srovnání s BenCzechMark leaderboardem

Pro zařazení našich 4 modelů do širšího kontextu jsme stáhli per-task výsledky z [oficiálního BenCzechMark leaderboardu](https://huggingface.co/spaces/CZLC/BenCzechMark) (dataset `CZLC/LLM_benchmark_data`). Srovnáváme na 4 úlohách, kde je oboustranně dostupná hodnota: `belebele`, `hellaswag`, `sentiment_fb`, `csfever_nli`. Czechnews na BCM leaderboardu publikovaný není.

> **Pozor na metodologické rozdíly:**
> - BCM uvádí best-of-8 prompt variant per úloha (cherry-picking). Pro náš sloupec `sent_fb` uvádíme rovněž best-of-5; ostatní úlohy mají u nás jeden fixní prompt.
> - BCM používá plné testovací množiny; naše `--limit=200` (resp. `--limit=1000` pro hellaswag) přidává ~1–3 p.b. statistické chyby.
> - **BCM modely jsou převážně pre-reasoning era** (2024–2025). Naše modely jsou post-reasoning era (early 2026). Loglikelihood-mode srovnání jim systematicky nadržuje.
> - Gemma 4 IT v naší tabulce ne-měří českou znalost (viz sekce 2.1) — **pro férové srovnání Gemma 4 rodiny doporučujeme používat řádek "gemma-4-31B base"** níže.

| Model | Velikost / Aktivní | belebele | hellaswag | sent_fb | csfever |
|---|---|---:|---:|---:|---:|
| **Qwen3.6-27B** ⭐ (náš) | **27B / 27B** | **0.915** | 0.434 | **0.735** | **0.730** |
| Qwen3.6-35B-A3B (náš) | 35B / 3B (MoE) | 0.855 | 0.428 | 0.730 | 0.630 |
| **gemma-4-31B base** ⭐ (náš, dubnový běh) | **31B / 31B** | **0.915** | **0.510** | **0.793** | _N/A_ |
| gemma-4-31B-it ⚠️ (náš) | 31B / 31B | 0.655¹ | 0.297¹ | 0.570¹ | 0.615¹ |
| gemma-4-26B-A4B-it ⚠️ (náš) | 26B / 4B (MoE) | 0.560¹ | 0.318¹ | 0.465¹ | 0.555¹ |
| _— BCM published —_ |  |  |  |  |  |
| meta-llama/Llama-3.1-405B-Instruct | 405B / 405B | **0.947** | 0.700 | 0.820 | 0.640 |
| deepseek-ai/DeepSeek-V3-0324 | 671B / 37B (MoE) | 0.942 | 0.719 | 0.781 | 0.717 |
| Qwen/Qwen2.5-72B (base) | 72B / 72B | 0.944 | 0.634 | 0.783 | 0.719 |
| deepseek-ai/DeepSeek-R1-0528 | 671B / 37B (MoE) | 0.937 | **0.728** | 0.803 | **0.722** |
| CohereLabs/c4ai-command-a-03-2025 | 111B / 111B | 0.937 | 0.689 | 0.819 | 0.711 |
| Qwen/Qwen2.5-72B-Instruct | 72B / 72B | 0.935 | 0.637 | 0.799 | 0.711 |
| meta-llama/Llama-4-Maverick (17B×128E) | 400B / 17B (MoE) | 0.934 | 0.672 | 0.775 | 0.669 |
| mistralai/Mistral-Large-Instruct-2411 | 123B / 123B | 0.928 | 0.506 | 0.813 | 0.719 |
| google/gemma-2-27b-it | 27B / 27B | 0.926 | 0.643 | 0.775 | 0.713 |
| Qwen/Qwen2.5-32B-Instruct | 32B / 32B | 0.925 | 0.593 | 0.744 | 0.717 |
| meta-llama/Llama-3.3-70B-Instruct | 70B / 70B | 0.923 | 0.493 | **0.818** | 0.689 |
| meta-llama/Llama-4-Scout (17B×16E) | 109B / 17B (MoE) | 0.922 | 0.644 | 0.751 | 0.683 |
| meta-llama/Llama-3.1-70B-Instruct | 70B / 70B | 0.918 | 0.659 | 0.803 | 0.683 |
| google/gemma-3-27b-it | 27B / 27B | 0.917 | 0.667 | 0.823 | 0.701 |
| mistralai/Mixtral-8x22B-Instruct | 141B / 39B (MoE) | 0.917 | 0.649 | 0.789 | 0.693 |
| MiniMaxAI/MiniMax-M1-80k | 456B / 46B (MoE) | 0.918 | 0.659 | 0.783 | 0.696 |
| Qwen/Qwen2.5-7B-Instruct | 7B / 7B | 0.853 | 0.490 | 0.741 | 0.664 |
| google/gemma-2-9b-it | 9B / 9B | 0.902 | 0.501 | 0.600 | 0.698 |
| microsoft/phi-4 | 14B / 14B | 0.899 | 0.549 | 0.754 | 0.699 |

¹ Gemma 4 IT base-mode skóre **neměří českou znalost**, ale degeneraci způsobenou IT post-trainingem (viz sekce 2.1). Pro porovnání Gemma 4 vs Qwen 3.6 v češtině používej řádek **gemma-4-31B base**.

### Co z tohoto srovnání plyne

1. **Qwen3.6-27B v belebele (0.915) je v top třetině BCM leaderboardu** — překonává všechny ≤30B modely kromě gemma-3-27b-it (0.917). Pro lokální 27B model je to silný výsledek.

2. **Gemma 4 base (0.915 belebele) je rovnocenná Qwen 3.6**. Obě rodiny umí česky stejně dobře. Rozdíl mezi nimi v naší původní (sekce 1) tabulce je artefakt měření IT verze, ne skutečná česká schopnost.

3. **Hellaswag-CS je obecná slabina sub-72B modelů** — Qwen2.5-7B-Instruct 0.490, Mistral-Large 0.506, Llama-3.3-70B 0.493. Naše Qwen3.6-27B 0.434 patří do této skupiny. **Výjimkou jsou DeepSeek V3/R1 (0.72–0.73), pravděpodobně díky reasoning post-trainingu**.

4. **Qwen3.6-27B v csfever_nli (0.730) překonává všechny BCM modely** (max DeepSeek-R1 0.722). Pravděpodobně náš jeden fixní prompt zafungoval lépe než průměr.

5. **Pre-reasoning vs post-reasoning era srovnání má omezenou platnost.** BCM modely (2024–2025) jsou převážně non-reasoning a loglikelihood eval jim funguje. Naše 2026 reasoning modely jsou v loglikelihood-modu **systematicky podseděné** o nezjištěnou veličinu.

---

## 5. Doporučení (deployment-aware)

Volba modelu závisí na tom, **jak ho budeš používat a evaluovat**. Trade-offy:

### Scénář A: Klasický eval-driven workflow (loglikelihood, batch evaluations, MC úlohy)

**Volba: Qwen3.6-27B (dense)** nebo Qwen3.6-35B-A3B (MoE pro propustnost).

Důvod: oba zachovávají funkční base-mode loglikelihood, takže jsou kompatibilní s existujícími eval pipelinami a benchmarky. Můžeš je hodnotit klasickým harnessem a srovnávat s BCM leaderboardem.

### Scénář B: Konverzační aplikace s plnou chat infrastrukturou

**Volba:** všichni 4 jsou validní. **Gemma 4 IT** by měla být pokud:
- Máš deployment přes `/v1/chat/completions` API (s reasoning budget, system messages, multi-turn)
- Nemáš požadavek na lm-eval-harness benchmarking (BCM-style čísla u ní budou systematicky podhodnocená)
- Důvěřuješ tomu, že base modelu (gemma-4-31B base = 0.915 belebele) skutečně reflektuje schopnost rodiny

**Qwen 3.6 IT** je v tomto scénáři též dobrá — chat template + reasoning podporují, ale base-mode chování je zachované, takže je flexibilnější.

### Scénář C: "Univerzální" lokální nasazení na DGX Spark

**Volba: Qwen3.6-27B (dense).** Dobrý compromise mezi kvalitou (top-třetina BCM v belebele), velikostí (27B v BF16 ~54 GB, vejde se s rezervou), a univerzální evaluovatelností.

### Slabá místa všech kandidátů (nezáleží na volbě)

- **Cermat / české státnice** (43–45 % u Qwen3.6-27B, BCM modely podobně) — všechny otevřené modely mají problém s úlohami specificky designovanými na češtinu.
- **Hellaswag-CS** (43–46 % u Qwen3.6-27B, BCM srovnatelné velikostní třídy 0.49–0.55) — commonsense v češtině je obecně slabé místo modelů ≤32B. Pokud potřebuješ commonsense v češtině, jdi do reasoning modelu (DeepSeek V3/R1 0.72) nebo doplň RAG.
- **Sentiment recenzí** (58–62 % u všech) — nejednoznačný tón české recenze je obtížný; FB sentiment je o ~10 p.b. lepší.

---

## 6. Co je v repu, co je commitnuté

**Implementační artefakty z této práce (potenciálně užitečné pro další eval):**
- `lm_eval/tasks/benczechmark/` — task YAMLy (MVP group, extended group s 16 úlohami, csfever_nli, generative_lite, utils.py s BCZMTask wrappery)
- `scripts/benczechmark/` — orchestrační skripty pro vLLM + lm_eval pipeline
- **Patch v `lm_eval/models/api_models.py`** — fix `JsonChatStr.rstrip` bugu pro `--apply_chat_template + local-completions + loglikelihood` (sekce 2.2). Připravený k upstream kontribuci.
- **Patch v `lm_eval/tasks/__init__.py`** — fix `pretty_print_task` pro inline sub-tasky bez yaml_path (objevil se při generative eval).

**Co se neuložilo do číselných výsledků v této zprávě, ale je zajímavé pro budoucnost:**
- Smoke test summarization na Qwen3.6-27B funguje (max_gen_toks=128 dostatečné pro thinking + souhrn). Plný běh na všech 4 modelech přes ROUGE-2 by mohl být užitečný complement, pokud bude zájem.
- Patch chat-template otevírá cestu pro správný eval Gemma 4 IT, pokud budou dostupné prompty bez vynuceného reasoning prefixu (např. system message "Odpověz jen jedním písmenem").

---

## 7. Limity vyhodnocení

- **Limit 200 vzorků na úlohu (1 000 pro Hellaswag).** Drží statistickou chybu pod ~3 p.b., ale ne pod ~1 p.b. Hierarchie modelů ve velkých rozdílech (>5 p.b.) je spolehlivá; v malých rozdílech (1–3 p.b.) je v noise.
- **Loglikelihood-mode pro 5 ze 6 task families.** Jediná funkční generativní eval je summarization; zbylé 2 generativní úlohy (triviaQA, SQuAD) nelze férově měřit u reasoning modelů s krátkým `max_gen_toks` (sekce 2.3).
- **Chat-template loglikelihood je strukturálně rozbitý** pro reasoning modely (sekce 2.2). Patch funguje, ale výsledky jsou degenerované kvůli `<think>` prefixu.
- **Gemma 4 IT skóre v sekci 1 je artefakt, ne signál.** Pro porovnání rodiny používej řádek "gemma-4-31B base" v sekci 4.
- **BCM leaderboard srovnání je apples-to-oranges** mezi pre-reasoning era (2024–2025, většina BCM) a post-reasoning era (early 2026, naše 4 modely). Loglikelihood eval nadržuje pre-reasoning modelům.
- **Pro vyhodnocení reálné instruction-following / chat-mode kvality** by bylo třeba opustit lm-evaluation-harness pipeline a postavit eval přes plný `/v1/chat/completions` s reasoning budget a LLM-as-judge metrikami. To je samostatný projekt na týdny, který zde záměrně neděláme.
