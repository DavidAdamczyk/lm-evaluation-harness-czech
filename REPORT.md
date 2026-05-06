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

## Limity vyhodnocení

- Limit 200 vzorků na úlohu (10 000 pro Hellaswag) — drží statistickou chybu pod ~3 p.b., ale ne pod ~1 p.b.
- Loglikelihood-mode pouze; generative úlohy (sumarizace, otevřené QA) nejsou v této sadě.
- Instruct verze Gemma 4 jsou hodnoceny v base-mode (bez chat-template). Reálný uživatelský zážitek s chat-templatem může být lepší, ale rozdíl by musel být extrémní (~25 p.b.), což literatura pro Gemma 2/3 nepotvrzuje.
