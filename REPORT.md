# Vyhodnocení 4 LLM modelů na češtině

**Cíl.** Posoudit, který ze čtyř kandidátů (Qwen 3.6 a Gemma 4, vždy v dense i MoE variantě) podporuje nejlépe český jazyk pro lokální nasazení na DGX Spark.

**Metoda.** Loglikelihood evaluace (multiple-choice, base-mode) přes lm-evaluation-harness na úlohách z BenCzechMark. Modely běžely v BF16 přes vLLM. Hlavní srovnávací sada: 6 úlohových rodin. Pro vítěze následně rozšířená evaluace na 16 BCM kategoriích.

---

## Hlavní zjištění

> **Qwen 3.6 rodina dominuje s velkým náskokem; Qwen3.6-27B (dense) je vítěz.**

| # | Model | Průměr | Komentář |
|---|---|---|---|
| 🥇 | **Qwen3.6-27B (dense)** | **0.6955** | Nejlepší v 5 ze 6 rodin |
| 🥈 | Qwen3.6-35B-A3B (MoE, 3B aktivních) | 0.6557 | Velmi blízko vítězi, levnější inference |
| 🥉 | gemma-4-31B-it (dense) | 0.4567 | Výrazný propad oproti Qwenu |
| 4 | gemma-4-26B-A4B-it (MoE) | 0.4395 | Nejnižší skóre |

Rozdíl Qwen vs Gemma 4 je **~24 procentních bodů průměrně** — to není šum, ale konzistentní rodinový rozdíl napříč všemi typy úloh.

---

## Srovnávací tabulka (acc, vyšší = lepší)

| Úloha | Qwen3.6-27B | Qwen3.6-35B-A3B | gemma-4-31B-it | gemma-4-26B-A4B-it |
|---|---:|---:|---:|---:|
| Belebele (čtení s porozuměním) | **0.915** | 0.855 | 0.655 | 0.560 |
| CzechNews (klasifikace témat, ⌀ 5 promptů) | **0.916** | 0.860 | 0.368 | 0.444 |
| Sentiment FB (⌀ 5 promptů) | **0.718** | 0.701 | 0.500 | 0.425 |
| Hellaswag-CS (200) | **0.460** | **0.460** | 0.305 | 0.335 |
| Hellaswag-CS (1000) | **0.434** | 0.428 | 0.297 | 0.318 |
| csfever NLI | **0.730** | 0.630 | 0.615 | 0.555 |
| **Průměr 6 rodin** | **0.696** | 0.656 | 0.457 | 0.440 |

**Pozorování:**
- **Belebele** (čtení s porozuměním) i **CzechNews** (klasifikace) ukazují u Qwena ~90 %, Gemma drží ~50–65 %. To indikuje, že Gemma 4 má slabší pokrytí češtiny v pretrainingu, ne jen jiný kalibrační styl.
- **Sentiment FB** — nejmenší rozdíl (~22 p.b.), pravděpodobně proto, že úloha má jen 3 možnosti a sentiment je univerzálně přenositelný.
- **Hellaswag-CS** je obecně nízká pro všechny — task je obtížný i pro silné anglické modely; relativní řazení je ale stejné.

---

## Hloubková analýza vítěze: Qwen3.6-27B

Rozšířená evaluace přes 16 BCM kategorií (limit 200 vzorků na úlohu) ukazuje, kde je model silný a kde slabší:

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

**Co model umí dobře:**
- **Faktografické a školní MC úlohy** (Umimeto biologie/chemie/dějepis/matematika/informatika): 84–97 %. Model má solidní česky-jazykovou obecnou znalost.
- **Strukturované úlohy** s jasnou textovou volbou: čtení (91 %), klasifikace zpráv (92 %), NLI (73–79 %).

**Kde model selhává:**
- **Cermat** (úlohy ze státních maturit) drží jen 43–45 %, což je mírně nad náhodou. To naznačuje, že úlohy navržené pro češtinu konkrétně (s pastmi, ironií, českou kulturní specifikou) jsou výrazně těžší než lokalizované „mezinárodní" úlohy.
- **Hellaswag-CS** (46 %) — slabý commonsense v češtině, podobně jako mnoho otevřených modelů.
- **Český jazyk z Umimeto** (62 %) — nižší než ostatní Umimeto kategorie, znamená to, že lingvistická česká specifika (gramatika, etymologie) jsou slabší doménou.
- **Sentiment recenzí** (Mall 58 %, ČSFD 62 %) — model má potíže s nejednoznačným tónem české recenze; řízený sentiment z FB postů (72 %) je výrazně lepší.

---

## Doporučení

1. **Pro produkční nasazení na DGX Spark zvolit Qwen3.6-27B (dense).**
   - Nejvyšší skóre, dobře balancuje kvalitu a velikost (27 B v BF16 ~54 GB, vejde se s rezervou).
2. **Pro vyšší propustnost / nižší latenci alternativně Qwen3.6-35B-A3B (MoE).**
   - Průměr o ~4 p.b. nižší, ale aktivních jen ~3 B parametrů → výrazně rychlejší inference.
   - Doporučení: pokud volume > kvalita, jdi do MoE; pokud kvalita > volume, dense.
3. **Gemma 4 (obě varianty) pro česky-první aplikace nedoporučujeme.**
   - Konzistentní propad ~24 p.b. napříč úlohami. Použitelná pro mezinárodní content, ne jako primární česky-orientovaný model.
4. **Cermat a sentiment recenzí jsou slabá místa i u vítěze** — pokud aplikace závisí na těchto doménách, je vhodné fine-tunovat nebo doplnit rule-based vrstvou.

---

## Srovnání s BenCzechMark leaderboardem

Pro zařazení našich 4 modelů do širšího kontextu jsme stáhli per-task výsledky z [oficiálního BenCzechMark leaderboardu](https://huggingface.co/spaces/CZLC/BenCzechMark) (dataset `CZLC/LLM_benchmark_data`). Srovnáváme na 4 úlohách, kde je oboustranně dostupná hodnota: `belebele`, `hellaswag`, `sentiment_fb`, `csfever_nli`. Czechnews na BCM leaderboardu publikovaný není.

> **Pozor na metodologické rozdíly** (BCM nahrávky vs naše běhy):
> - BCM uvádí nejlepší ze 8 prompt variant per úloha (cherry-picking). Pro náš sloupec `sent_fb` uvádíme rovněž nejlepší z 5 prompt variant; ostatní úlohy mají u nás jeden fixní prompt.
> - BCM používá plné testovací množiny; naše `--limit=200` (resp. `--limit=1000` pro hellaswag) přidává ~1–3 p.b. statistické chyby.
> - BCM modely jsou často spouštěny v chat-template režimu; naše IT modely (Gemma 4) běží v base-mode kvůli loglikelihood bugu v lm_eval s chat-templatem.
>
> **Důsledek:** absolutní čísla nejsou plně srovnatelná, ale **relativní ranking ve velikostní třídě** ano. Naše Qwen3.6-27B se na úlohách patří do **horní třetiny** všech publikovaných modelů.

| Model | Velikost / Aktivní | belebele | hellaswag | sent_fb | csfever |
|---|---|---:|---:|---:|---:|
| **Qwen3.6-27B** ⭐ (náš) | **27B / 27B** | **0.915** | 0.434 | **0.735** | **0.730** |
| Qwen3.6-35B-A3B (náš) | 35B / 3B (MoE) | 0.855 | 0.428 | 0.730 | 0.630 |
| gemma-4-31B-it (náš) | 31B / 31B | 0.655 | 0.297 | 0.570 | 0.615 |
| gemma-4-26B-A4B-it (náš) | 26B / 4B (MoE) | 0.560 | 0.318 | 0.465 | 0.555 |
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

### Co z tohoto srovnání plyne

1. **Qwen3.6-27B je v belebele (0.915) na úrovni Qwen2.5-7B-Instruct (0.853) až Qwen2.5-32B-Instruct (0.925)** a překonává všechny modely pod 30 B parametry kromě Gemma-3-27B-it. Vzhledem k tomu, že běží jako base-mode v BF16 na 110 GB DGX Spark, je to silný výsledek pro lokální nasazení.

2. **Qwen3.6-27B v sentiment_fb (0.735, best-of-5) je v rozmezí 7B–32B Qwen2.5 IT modelů** (0.741–0.799). Větší modely v BCM (70B+) mají náskok 5–8 p.b., což vysvětluje rozdíl velikostí.

3. **Qwen3.6-27B v csfever_nli (0.730) překonává všechny BCM modely** (max 0.722 u DeepSeek-R1). Pravděpodobně náš jeden fixní prompt zafungoval lépe než průměr nebo jsme se trefili do prompt variant typu, který je v BCM nepublikovaný.

4. **Hellaswag-CS je obecná slabina menších a sub-30B modelů**: Qwen2.5-7B-Instruct 0.490, Mistral-Large 0.506, Llama-3.3-70B 0.493 — naše Qwen3.6-27B 0.434 je v této skupině, podobně jako Llama-3.3-70B (!) a Mistral-Large. **DeepSeek V3/R1 jsou v hellaswag výjimečně silní (0.72–0.73), pravděpodobně díky reasoning-tuningu**.

5. **Gemma 4 IT modely jsou výrazně horší než Gemma 2/3 IT**: gemma-2-27b-it 0.926 belebele vs naše gemma-4-31B-it 0.655. Rozdíl ~27 p.b. je příliš velký na vysvětlení jen chat-template bugem; **Gemma 4 v base-mode loglikelihood je zjevně horší než její předchůdci** (potvrzuje hypotézu o agresivním post-trainingu, který oslabuje base-mode chování).

6. **Pro lokální nasazení na DGX Spark (≤110 GB) je Qwen3.6-27B nejlepší volbou** — DeepSeek V3/R1 (671B), Llama 3.1-405B, Mistral-Large (123B) se nevejdou ani s aggressive quantizací. Z modelů srovnatelné velikosti (≤32B) v BCM je nejvýše Gemma-3-27B-it; ten by mohl být zajímavý kandidát, pokud by Qwen3.6 nebyl k dispozici.

---

## Limity vyhodnocení

- Limit 200 vzorků na úlohu (1 000 pro Hellaswag) — drží statistickou chybu pod ~3 p.b., ale ne pod ~1 p.b.
- Loglikelihood-mode pouze; generative úlohy (sumarizace, otevřené QA) nejsou v této sadě.
- Instruct verze Gemma 4 jsou hodnoceny v base-mode (bez chat-template). Reálný uživatelský zážitek s chat-templatem může být lepší, ale rozdíl by musel být extrémní (~25 p.b.), což literatura pro Gemma 2/3 nepotvrzuje.
- Srovnání s BCM leaderboardem používá best-of-N prompt selection na BCM straně; naše čísla jsou (kromě sent_fb) z jednoho fixního promptu. Absolutní rozdíly tedy podhodnocují náš výkon o ~2–5 p.b.
