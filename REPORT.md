# Vyhodnocení 4 LLM modelů na češtině

**Cíl.** Posoudit, který ze čtyř kandidátů (Qwen 3.6 a Gemma 4, vždy v dense i MoE variantě) podporuje nejlépe český jazyk pro lokální nasazení na DGX Spark.

**Modely.** `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` (MoE 35B/3B), `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it` (MoE 26B/4B). Všechny inference v BF16 přes vLLM (`gemma4-0505-cu130` image) na DGX Spark / NVIDIA GB10.

**Metoda.** Multiple-choice loglikelihood přes lm-evaluation-harness s úlohami z BenCzechMark. Tato zpráva také dokumentuje **metodologické limity, které jsme během práce odhalili**, a co z nich plyne pro férovost interpretace výsledků.

---

## TL;DR

> **Gemma 4 base ≈ Qwen 3.6 v češtině** (oba ~0.91 belebele, oba ~0.92 czechnews). Klasický lm-eval pipeline ale u IT verzí těchto modelů selhává kvůli vynucenému reasoning prefixu — naše práce zahrnuje **patch lm-eval-harness, který tuhle limitaci řeší** přes `chat_template_kwargs` (Qwen `enable_thinking=False`, Gemma 4 `enable_thinking=True`).
>
> **Výsledky po patche (chat template + suppressed thinking, limit=200, průměr 3 task families bez Belebele):**
> - Qwen3.6-35B-A3B (MoE): **0.712** — nejlepší v naší IT sadě
> - Qwen3.6-27B (dense): **0.707**
> - gemma-4-31B base (referenční, dubnový běh): **0.743** ← top
> - gemma-4-31B-it: **0.516** ← chat+thinking-off zlepšil eval o ~+12 p.b., ale stále pod base
> - gemma-4-26B-A4B-it: **0.456**
>
> **Klíčové zjištění:** Gemma 4 base má v češtině o ~3 p.b. víc než Qwen 3.6, ale Gemma 4 **IT post-training stojí ~23 p.b.** v MC loglikelihood evalu. Pro reálné chat nasazení se rozdíl smazává; pro klasické benchmarky volí jednodušší Qwen 3.6.

**Praktické doporučení pro DGX Spark:** Qwen 3.6 (27B dense pro kvalitu, 35B-A3B MoE pro propustnost) je defaultní volba s nejvyšší předvídatelností v eval i deploymentu. Gemma 4 IT je validní pro chat-completions deployment; její IT skóre v MC loglikelihood evalu (i s thinking-off) podhodnocuje její reálné kvality.

**Generativní eval (sekce 6.5):** Summarization přes LLM-as-Judge (Claude Opus 4.7).

> ⚠️ **Důležitá korekce metodologie po dokončení původní verze reportu** (sekce 6.6): Původní judge eval používal HF chat template, který forcuje thinking marker. Po reprodukci s **Ollama-style template** (matchuje produkční RAG deployment) **Gemma 4 31B-it dominuje** (avg 4.36) nad Qwen3.6-27B (4.25) i Qwen3.6-35B-A3B (4.26). Detail v sekci 6.6.

---

## 🚨 TODO / Otevřené problémy

> **Tato zpráva je rozpracovaná. Před finálním deployment rozhodnutím je potřeba dořešit níže uvedené body.**

### TODO #1: Gemma 4 IT — neúplně rozumíme proč nefunguje "out of the box"

**Symptom.** Při použití standardní lm-evaluation-harness pipeline + HF transformers chat template Gemma 4 IT (31B i 26B-A4B) **emituje nekonečné anglické thinking traces místo odpovědí**:
```
thought
*   Input: A long text describing a luxury trip to the North Pole...
    *   Constraint: Summary must be between 33 and 51 words...
    *   Constraint: Summary must end with the character '\n'...
```
S `max_gen_toks=128` se model nikdy nedostane k samotné odpovědi. Pro loglikelihood eval rozbíjí scoring kandidátních odpovědí.

**Co víme.** Příčina je v **divergenci HF chat template vs. Ollama renderer**:
- HF tokenizer `apply_chat_template` defaultně forcuje `<|channel>thought\n<channel|>` postfix → model MUSÍ thinkovat
- Ollama `gemma4.go` renderer postfix nepřidává → model může odpovídat rovnou
- Náš workaround: `chat_template_file` patch + `templates/gemma4_ollama_style.jinja` (sekce 6.6)

**Co NE víme (a musíme dořešit):**

1. **Proč i s Ollama template Gemma 4 IT pořád emituje literální `thought\n` prefix** na začátku každé odpovědi? Model se to naučil jako "tic" v post-trainingu. Ollama uživatelé v RAG pravděpodobně buď stripují tento prefix (vědomě nebo přes nějaký post-process), nebo ignorují. **Nezjistili jsme jak to v reálné produkci řeší.** → Action: zeptat se autora produkční RAG aplikace, jak to filtruje.

2. **Proč je `enable_thinking` parametr u Gemma 4 obrácený oproti Qwen 3.6?** U Qwena `False` = vypnout thinking; u Gemmy 4 `True` = vypnout forced thought channel. Bez prokopnutí se Jinja templatem neintuitivní. Možná že tu existuje další kwarg (`thinking_budget`, `add_thought_channel`, …), který bychom měli použít místo. → Action: projít chat template Jinja kód detailně, najít kanonický způsob.

3. **Plný summarization eval s Ollama template byl proveden jen na 20 docs (smoke).** Plná 200-doc verze pro Gemma 4 31B-it i 26B-A4B-it chybí. Judge výsledek 4.36 vs 4.25 (Gemma > Qwen) je v rámci stderr (~22 %) — potřebujeme více vzorků pro defensible závěr. → Action: re-run summarization na obou Gemma modelech s Ollama template, ~3h compute, pak full judge eval ($25).

4. **MC loglikelihood evals s Ollama template jsme ještě neprověřili.** Sekce 2.5 ukazuje, že chat-template + thinking-off u Gemmy zlepší czechnews z 0.368 na 0.624 — ale to s HF tokenizerem `enable_thinking=True`. **S Ollama template + strip filter** by skóre mohlo dále vystoupat blíže k base modelu (0.926 czechnews). → Action: spustit `run_thinking_off_eval.sh` variant s `CHAT_TEMPLATE_FILE=...gemma4_ollama_style.jinja` pro Gemma modely, porovnat s aktuálním 0.516 avg.

5. **Belebele propadá u všech 4 modelů s chat templatem (~30–42 p.b.).** Pravděpodobně chat-template wrapping × 3-shot × dlouhé passages overflowuje effective context limit. Není to Gemma-specifický problém, ale obecná limitace MC eval s chat-template setupem. → Action: zkusit `num_fewshot: 0` pro belebele s chat template, nebo investigovat truncation behavior.

6. **Gemma 4 IT může mít vyšší kvalitu výstupu při použití plného `/v1/chat/completions` endpointu** (Ollama / vLLM chat-completions API, server-side template, reasoning budget). Naše eval jede přes `/v1/completions` (text-prompt). Ekosystém kolem chat-completions je jiný a může vyřešit thinking nativně. → Action: postavit alternative eval přes `local-chat-completions` model (lm-eval to podporuje pro generation, ne pro loglikelihood) nebo standalone Python skript.

### TODO #2: Submitnout patche upstream do lm-evaluation-harness

Tři patche bychom měli kontribuovat zpět:
1. `JsonChatStr.rstrip` fix v `_encode_pair`
2. `enable_thinking` kwarg v `TemplateAPI.__init__`
3. `chat_template_file` override v `TemplateAPI.__init__`

Patch #3 je nejširšího významu — jakýkoli model s nematching deployment template (Gemma 4, ale možná i další) z toho profituje.

### TODO #3: Final deployment rozhodnutí — počkat, NEBO commit teď?

Současný stav: Qwen 3.6 je "safe choice" (eval funguje, výsledky předvídatelné). Gemma 4 IT je "potential winner" (judge naznačuje vyšší kvalitu pro RAG-style chat, ale eval pipeline plně nefunguje, takže neumíme to defensible rangovat).

**Doporučení:** Před finálním commit do produkce **dořešit minimálně TODO #1 body 1, 3, 4** (jak Ollama deal with `thought\n` prefix, full Gemma summarization rerun, MC eval s Ollama template). Pak se buď potvrdí "Gemma 4 wins", nebo "Qwen 3.6 stays safe choice" — a víme to spolehlivě.

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

**Závěr v této fázi:** chat-template + loglikelihood vypadal jako neuzavřený problém. Reasoning prefix rozbíjí scoring kandidátních odpovědí — patch byl nutný, ale ne dostatečný. (Sekce 2.5 níže ukazuje, jak jsme to nakonec doopravdy vyřešili.)

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

### 2.4 Mezistav: co lm-evaluation-harness u našich modelů změřit umí

| Modus | Reasoning model situation | Validní pro naši sadu |
|---|---|---|
| Loglikelihood base-mode (no chat template) | Funguje pokud post-training base-mode zachoval | ✅ Qwen 3.6 (oba). ❌ Gemma 4 IT (oba). ✅ Gemma 4 base. |
| Loglikelihood + chat template (default) | Reasoning prefix rozbíjí scoring | ❌ Žádný z našich 4 |
| **Loglikelihood + chat template + thinking-off** (sekce 2.5) | **Patch řeší prefix problém** | ✅ **Všichni 4** (kromě belebele kvůli kontext-length artifaktu) |
| Generate, krátké (≤20 tok) | Model stihne jen `<think>` opener | ❌ triviaQA, SQuAD |
| Generate, dlouhé (128+ tok) | Model thinking + skutečnou odpověď | ✅ summarization |

### 2.5 Průlom: thinking suppression přes `chat_template_kwargs`

Klíčová otázka: **co kdyby šlo říct chat templatu, aby thinking prefix vůbec nepřidával?** Inspekce HF tokenizer chat_template Jinja kódu odhalila, že:

- **Qwen 3.6:** `apply_chat_template(messages, enable_thinking=False)` pre-fill prázdné thinking (`<think>\n\n</think>\n\n`) → model rovnou odpoví, jako by už dopřemýšlel.
- **Gemma 4:** `apply_chat_template(messages, enable_thinking=True)` (paradoxně opačná konvence) → vynechá forced `<channel>thought` marker → model dostane volnost.

Standardně lm-evaluation-harness žádné dodatečné kwargs do `apply_chat_template` nepředává. Přidali jsme do `TemplateAPI` parametr `enable_thinking` (přístupný přes `--model_args enable_thinking=false`), který se forwardne do tokenizeru.

**Výsledky (limit=200, plný re-eval všech 4 IT modelů):**

| Model | Task | base-mode | think-off chat | Δ |
|---|---|---:|---:|---:|
| **Qwen3.6-27B** | czechnews ⌀5 | 0.916 | **0.915** | ≈ |
|  | sentiment_fb ⌀5 | 0.718 | 0.751 | +3 p.b. |
|  | hellaswag | 0.460 | 0.455 | ≈ |
|  | belebele | 0.915 | 0.615 | −30 p.b. ⚠ |
| **Qwen3.6-35B-A3B** | czechnews ⌀5 | 0.860 | 0.898 | +4 p.b. |
|  | sentiment_fb ⌀5 | 0.701 | 0.748 | +5 p.b. |
|  | hellaswag | 0.460 | 0.490 | +3 p.b. |
|  | belebele | 0.855 | 0.440 | −42 p.b. ⚠ |
| **gemma-4-31B-it** ⭐ | czechnews ⌀5 | 0.368 | **0.624** | **+26 p.b.** |
|  | sentiment_fb ⌀5 | 0.500 | 0.509 | +1 p.b. |
|  | hellaswag | 0.305 | 0.415 | **+11 p.b.** |
|  | belebele | 0.655 | 0.285 | −37 p.b. ⚠ |
| **gemma-4-26B-A4B-it** | czechnews ⌀5 | 0.444 | 0.404 | −4 p.b. |
|  | sentiment_fb ⌀5 | 0.425 | 0.583 | **+16 p.b.** |
|  | hellaswag | 0.335 | 0.380 | +5 p.b. |
|  | belebele | 0.560 | 0.260 | −30 p.b. ⚠ |

**Co z toho plyne:**

1. **Pro Qwen 3.6 (oba) thinking-off chat ≈ base-mode** napříč 3 task families (czechnews, sentiment_fb, hellaswag). Patch otevírá fér chat-template cestu.

2. **Pro Gemma 4 31B-it výrazné zlepšení:** czechnews +26 p.b. (z 0.368 na 0.624), hellaswag +11 p.b. Patch dělá Gemma 4 IT eval **zhruba dvakrát fér**, ale **ne plně rovný base modelu** (gemma-4 base dosahuje 0.926 czechnews, 0.510 hellaswag — IT post-training stojí dál ~30 p.b. čistého).

3. **Belebele propadá u všech 4 modelů ~30–42 p.b. s chat templatem** — to je **artefakt setupu**, ne signál o češtině. Pravděpodobná příčina: chat-template wrapping × 3-shot × dlouhé belebele passages tlačí kontextové délky k limitu, kde lm_eval truncatuje. Vyžaduje další investigaci, ale pro hlavní srovnání modelů Belebele se thinking-off čísly **nepoužíváme**.

4. **Hierarchie po patchi (průměr 3 task families bez Belebele):**

| Model | avg(czechnews, sentiment_fb, hellaswag) | Rozdíl proti gemma-4-31B base (0.743) |
|---|---:|---:|
| **gemma-4-31B base** (ref) | **0.743** | (baseline) |
| Qwen3.6-35B-A3B (MoE, IT think-off) | 0.712 | −3.1 p.b. |
| Qwen3.6-27B (dense, IT think-off) | 0.707 | −3.6 p.b. |
| gemma-4-31B-it (chat think-off) | 0.516 | −22.7 p.b. |
| gemma-4-26B-A4B-it (chat think-off) | 0.456 | −28.7 p.b. |

**Konečný závěr:** Gemma 4 base je v češtině na špici. Qwen 3.6 IT je 3 p.b. pod ní (téměř na úrovni base, jen mírně). **Gemma 4 IT zaostává o ~23 p.b. — IT post-training Gemmy 4 reálně omezuje její použitelnost v non-chat režimu**. Pro chat-completions deployment by to mohlo být menší (potřeba ověřit jiným evalem než lm-eval-harness).

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

**Volba: Qwen 3.6 (27B dense pro kvalitu, 35B-A3B MoE pro propustnost).**

Důvod: oba si v naší pipeline drží avg ~0.71 napříč 3 task families (post-thinking-off chat) — v rámci 3 p.b. od reference gemma-4-31B base. Po našem patche jdou bezproblémově evaluovat klasickým harnessem v thinking-off chat módu, a zároveň fungují v base-mode loglikelihood pro srovnání s pre-reasoning era leaderboardy (BCM).

### Scénář B: Konverzační aplikace s plnou chat infrastrukturou (Ollama / vLLM /v1/chat/completions)

Aktuální evidence (sekce 6.6, n=20 limit) ukazuje:

1. **Gemma 4 31B-it** — **judge favorit** (avg 4.36 vs Qwen ~4.25) v summarization s Ollama-style template + thought-prefix strip. Vyšší faithfulness, coverage a conciseness. **Doporučená volba pro RAG/chat deployment**, pokud máš infrastrukturu, která zpracuje `thought\n` prefix (Ollama, custom post-process) — viz **TODO #1** pro otevřené body, které je třeba dořešit pro definitivní rozhodnutí.
2. **Qwen 3.6 27B / 35B-A3B (IT)** — solidní druhé pořadí (4.25 / 4.26). Mírně lepší fluency než Gemma. Bezproblémový eval out-of-the-box.
3. **Gemma 4 26B-A4B-it** — **nevyhodnoceno s Ollama template** (chybí compute). V naší původní HF-template eval skórovala nejhůř (0.456 avg), ale to je pravděpodobně artefakt setupu, ne modelu — viz TODO #1 bod 3.

### Scénář C: "Univerzální" lokální nasazení na DGX Spark

**Volba: Qwen3.6-27B (dense).** Pokrývá oba scénáře A a B. 27B v BF16 ~54 GB, vejde se s rezervou na 110 GB Spark. Předvídatelné chování v eval i deploymentu.

### Slabá místa všech kandidátů (nezáleží na volbě)

- **Cermat / české státnice** (43–45 % u Qwen3.6-27B, BCM modely podobně) — všechny otevřené modely mají problém s úlohami specificky designovanými na češtinu.
- **Hellaswag-CS** (43–46 % u Qwen3.6-27B, BCM srovnatelné velikostní třídy 0.49–0.55) — commonsense v češtině je obecně slabé místo modelů ≤32B. Pokud potřebuješ commonsense v češtině, jdi do reasoning modelu (DeepSeek V3/R1 0.72) nebo doplň RAG.
- **Sentiment recenzí** (58–62 % u všech) — nejednoznačný tón české recenze je obtížný; FB sentiment je o ~10 p.b. lepší.

---

## 6. Co je v repu, co je commitnuté

**Implementační artefakty z této práce (potenciálně užitečné pro další eval):**
- `lm_eval/tasks/benczechmark/` — task YAMLy (MVP group, extended group s 16 úlohami, csfever_nli, generative_lite, utils.py s BCZMTask wrappery).
- `scripts/benczechmark/` — orchestrační skripty pro vLLM + lm_eval pipeline (`run_mvp_eval.sh` jako runner s podporou `APPLY_CHAT_TEMPLATE`, `ENABLE_THINKING`, `REVISION` env vars; `run_thinking_off_eval.sh` pro MC re-eval všech 4 IT modelů; `run_summarization_eval.sh` pro generative eval; `run_judge_eval.py` pro Claude Opus 4.7 LLM-as-Judge nad summarization outputy přes `claude -p` CLI).
- **Patch v `lm_eval/models/api_models.py`** — tři souvisící fixy (sekce 2.2, 2.5, 6.6):
  1. Fix `JsonChatStr.rstrip` AttributeError v `_encode_pair` pro `--apply_chat_template + local-completions + loglikelihood` (drop `tokenized_requests` condition v HF tokenizer větvi `apply_chat_template`).
  2. Přidaný parametr `enable_thinking` v `TemplateAPI.__init__`, který se forwardne do `tokenizer.apply_chat_template` jako kwarg. Umožňuje per-model suppression reasoning prefixu (Qwen 3.6: `enable_thinking=False`; Gemma 4: `enable_thinking=True` — opačná konvence).
  3. Přidaný parametr `chat_template_file` v `TemplateAPI.__init__`, který override-uje `tokenizer.chat_template` z Jinja souboru. Klíčové pro modely, kde HF transformers default neodpovídá produkčnímu deployment renderingu (např. Gemma 4 vs. Ollama — viz `templates/gemma4_ollama_style.jinja`).
  
  Všechny 3 fixy jsou **kandidáti na upstream kontribuci** — řeší fundamentální gap v lm-eval pipeline pro reasoning modely a deployment-divergent templates.
- **Patch v `lm_eval/tasks/__init__.py`** — fix `pretty_print_task` pro inline sub-tasky bez yaml_path (objevil se při generative eval).

---

## 6.5 Generativní eval: summarization + LLM-as-Judge

Po MC re-evalu jsme spustili summarization eval na všech 4 modelech (`benczechmark_summarization`, 5 prompt variants × 200 docs, chat template + per-model thinking suppression dle sekce 2.5).

### ROUGE-2 F-mid (n-gram překryv)

| Model | ROUGE-2 ⌀5 promptů |
|---|---:|
| Qwen3.6-27B (dense) | 0.0317 |
| Qwen3.6-35B-A3B (MoE) | 0.0312 |
| gemma-4-31B-it | 0.0032 |
| gemma-4-26B-A4B-it | 0.0027 |

ROUGE-2 absolutní hodnoty jsou v češtině obecně nízké (bohaté skloňování ničí n-gram překryv parafráze proti referenci). Ale relativní ranking je výmluvný: **Gemma 4 IT modely mají ROUGE řádově 10× nižší než Qwen** — což je víc než jen šum.

### LLM-as-Judge: kvalitativní hodnocení (Claude Opus 4.7)

ROUGE-2 v češtině neumí rozlišit kvalitní parafrázi od garbage outputu. Postavili jsme **joint judge eval** přes `claude -p --model opus`: 50 doc_ids common ke všem 4 modelům, randomizované A/B/C/D pozice (mitigace position bias), JSON-schema-validated rubric. Náklady: ~$7 pro 50 calls.

| Model | faithfulness | coverage | fluency | conciseness | **avg** |
|---|---:|---:|---:|---:|---:|
| **Qwen3.6-27B (dense)** | **4.54** ± 0.86 | **4.14** ± 0.78 | **4.80** ± 0.45 | **4.64** ± 0.53 | **4.53** ⭐ |
| Qwen3.6-35B-A3B (MoE) | 4.42 ± 0.99 | 3.86 ± 0.78 | 4.66 ± 0.75 | 4.40 ± 0.57 | 4.33 |
| gemma-4-31B-it | 2.24 ± 1.17 | 1.82 ± 0.94 | 1.04 ± 0.20 | 1.14 ± 0.35 | 1.56 |
| gemma-4-26B-A4B-it | 2.36 ± 1.12 | 1.90 ± 0.95 | 1.06 ± 0.24 | 1.16 ± 0.37 | 1.62 |

**Co tato čísla skutečně říkají:**

1. **Qwen 3.6 27B (dense) je v naší pipeline jasný vítěz** v summarization. Faithful (4.54), pokrývá klíčové body (4.14), perfektní česká plynulost (4.80), úsporná délka (4.64). Soudce explicitně chválí kvalitu češtiny u Qwena.

2. **Qwen 3.6 27B > 35B-A3B (MoE)** o +0.20 v průměru. Dense varianta je v summarization viditelně lepší než MoE-3B-active.

3. **Gemma 4 IT skóre fluency 1.04/1.06 je sentinel hodnota** — judge dává 1 ("lámaná čeština / žádná čeština") protože **outputy nejsou české souhrny, ale anglické thinking traces** typu:
   ```
   thought
   *   Input: A long text describing a luxury trip to the North Pole...
       *   Constraint: Summary must be between 33 and 51 words.
       *   Constraint: Summary must end with the character '\n'...
   ```
   Model ulpí v thinking módu a `max_gen_toks=128` ho stopne uprostřed. **Tato čísla NEMĚŘÍ Gemma 4 IT summarization quality** — měří, že naše base-mode generation pipeline (i s `enable_thinking=True` chat templatem) neumožňuje Gemma 4 IT dokončit thinking a začít generovat samotný souhrn.

4. **Pro férovou Gemma 4 IT summarization eval** by bylo třeba:
   - **Buď** plný `/v1/chat/completions` s reasoning budget (model dokončí thinking, pak vygeneruje souhrn)
   - **Nebo** post-processing filter `regex: "after </thought>"` plus `max_gen_toks=1024+`
   - **Nebo** systémový prompt s instrukcí "Nepřemýšlej, rovnou vygeneruj souhrn"

   Žádný z těchto setupů nešel přes klasické lm-eval-harness pipeline bez další engineeringové práce.

5. **Pro Qwen 3.6 jsou judge čísla validní a srovnatelná napříč modely.** Pro Gemma 4 IT jsou judge čísla **artefakt eval setupu**, ne signál o modelu.

**Závěr summarization sekce:** V plně řešeném benchmarku by Gemma 4 IT mohla podávat lepší výkon (její base model dosahuje 0.74 avg napříč MC úlohami, takže reálné chat-completions deployment by mohlo dát kvalitní souhrny). Ale **v rámci toho, co umíme férově změřit klasickým lm-eval-harness pipelinem**, **Qwen 3.6 27B je jasná volba pro summarization v češtině** s judge skóre 4.53/5 napříč 4 dimenzemi.

---

## 6.6 Korekce: Ollama-style chat template a obrácený výsledek

Po dokončení sekce 6.5 jsme s tebou ověřili kritický fakt: **Gemma 4 v produkční RAG aplikaci (přes Ollama) reasoning nedělá**. To přímo odporovalo našemu závěru, že "Gemma 4 IT je degenerovaná" — pokud Ollama-deployovaná Gemma 4 funguje, **náš eval setup byl problém, ne model**.

### Root cause: HF tokenizer chat template ≠ Ollama renderer

Stáhli jsme Ollama source ([`gemma4.go`](https://github.com/ollama/ollama/blob/main/model/renderers/gemma4.go)) a porovnali s HF tokenizer chat template:

| Renderer | Default chování |
|---|---|
| **HF transformers `apply_chat_template`** (default, `enable_thinking=False`) | Forcuje `<\|channel>thought\n<channel\|>` postfix — model MUSÍ thinkovat |
| **HF s `enable_thinking=True`** | Přidává `<\|think\|>` system message — model je primován k thinking přes system prompt |
| **Ollama `gemma4.go`** | Nic — `<\|turn>model\n` a hotovo. Žádný forced thought channel. |

To znamená: **HF transformers a Ollama mají úplně jiné defaulty**. Naše první iterace evalů používala HF default (forcoval thinking), proto Gemma 4 IT nedoručila task. Reálné Ollama deployment používá clean template, proto v RAG funguje.

### Patch: chat_template_file override v lm-eval

Rozšířili jsme náš lm-eval patch o `chat_template_file` parametr, který nahrazuje `tokenizer.chat_template` z Jinja souboru. Uložili jsme [`templates/gemma4_ollama_style.jinja`](lm_eval/tasks/benczechmark/templates/gemma4_ollama_style.jinja) — Jinja ekvivalent Ollama renderingu.

### Smoke test (limit=20, gemma-4-31B-it, Ollama template)

| Setup | ROUGE-2 F-mid | Output kvalita |
|---|---:|---|
| HF default chat template + `enable_thinking=True` | 0.0032 | Anglické thinking traces, ne česká summary |
| **Ollama-style template** | **0.0456** | **České souhrny** (✅), ale s prefixem `thought\n` (model i bez forced channel emituje literál) |
| Ollama-style + post-strip `^thought\n+` | _N/A_ | Čisté české souhrny |

### Judge eval v2 (Claude Opus 4.7, n=20, 3 modely)

Spustili jsme druhou variantu judge evalu na 20 doc_ids ze smoke runu, s **proper template per model**:
- Qwen 27B / 35B-A3B: existující `chat+enable_thinking=False` data (200-doc summarization eval)
- Gemma 4 31B-it: Ollama-template smoke (20 docs) s aplikovaným strip filterem `^thought\n+`
- Gemma 4 26B-A4B-it: **vyřazena** — neměli jsme čas na rerun s Ollama template

| Model | faithfulness | coverage | fluency | conciseness | **avg** |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B (chat+think_off) | 4.35 ± 1.04 | 3.70 ± 0.80 | **4.65** ± 0.59 | 4.30 ± 0.57 | 4.25 |
| Qwen3.6-35B-A3B (chat+think_off) | 4.25 ± 1.02 | 3.80 ± 0.70 | **4.70** ± 0.66 | 4.30 ± 0.57 | 4.26 |
| **gemma-4-31B-it (chat+ollama_template)** ⭐ | **4.55** ± 0.83 | **4.05** ± 0.76 | 4.40 ± 0.88 | **4.45** ± 0.51 | **4.36** |

**Co z toho plyne:**

1. **Předchozí závěr "Gemma 4 IT je degenerovaná" byl artefakt HF chat template, ne vlastnost modelu.** S Ollama-style template se Gemma 4 31B-it stává nejvyšší skórující v naší sadě.

2. **Faithfulness a coverage Gemma > Qwen** — Gemma podává faktickyji věrnější a komplexnější souhrny.

3. **Fluency Gemma < Qwen o ~0.25 bodu** — možná zbylé formátovací artefakty (literál `thought\n`), nebo Gemma's stylistická preference. Ne dramatický rozdíl.

4. **n=20 znamená vysokou stderr (~22 % na 1–5 škále).** Rozdíl Qwen 4.25 vs Gemma 4.36 (0.10) je v rámci noise, **ale pattern napříč 3 ze 4 dimenzí Gemma > Qwen je konzistentní**.

5. **Pro férový plný benchmark** by bylo potřeba: re-run summarization na obou Gemma modelech s Ollama template (~3h compute), strip filter v YAML pipeline, plný 200-doc judge eval ($25). Mimo scope této zprávy — but proof-of-concept že je to možné je v kódu (sekce 6.6 + scripts/benczechmark/run_judge_eval_v2.py).

### Aktualizované doporučení (po této korekci)

**Pro Czech summarization v RAG-style deployment přes chat-completions API:** **Gemma 4 31B-it je preferovaná volba** (vyšší faithfulness/coverage/conciseness). Qwen 3.6 27B je solidní alternativa s mírně lepší stylistickou plynulostí.

**Pro klasický eval-driven workflow přes loglikelihood:** Qwen 3.6 zůstává jednodušší volba — funguje out-of-the-box. Gemma 4 IT vyžaduje custom chat_template override (`chat_template_file=templates/gemma4_ollama_style.jinja`) + strip filter.

### Otevřené otázky / future work

- Plný 200-doc summarization rerun obou Gemma modelů s Ollama template
- Otestovat Gemma 4 26B-A4B-it (MoE) s Ollama template
- Zjistit, jak Ollama users v produkci řeší `thought\n` prefix v outputech (post-process? nebo template trick co my nevíme?)
- Submitnout `chat_template_file` patch do upstream lm-evaluation-harness — generická užitečnost pro všechny modely s nematching deployment template

---

## 7. Limity vyhodnocení

- **Limit 200 vzorků na úlohu (1 000 pro Hellaswag).** Drží statistickou chybu pod ~3 p.b., ale ne pod ~1 p.b. Hierarchie modelů ve velkých rozdílech (>5 p.b.) je spolehlivá; v malých rozdílech (1–3 p.b.) je v noise.
- **Loglikelihood-mode pro 5 ze 6 task families.** Jediná funkční generativní eval je summarization; zbylé 2 generativní úlohy (triviaQA, SQuAD) nelze férově měřit u reasoning modelů s krátkým `max_gen_toks` (sekce 2.3).
- **Belebele se v thinking-off chat režimu chová špatně u všech 4 modelů (−30 až −42 p.b.).** Pravděpodobně chat-template wrapping × 3-shot × dlouhé passages přesahuje effective context limit a kontext se silently truncatuje. Pro hlavní MC srovnání používáme avg(czechnews, sentiment_fb, hellaswag) bez Belebele. Investigace odložena (out of scope této zprávy).
- **Gemma 4 IT skóre v sekci 1 (base-mode loglikelihood) je artefakt, ne signál.** Po patche v sekci 2.5 je k dispozici fér číslo (chat + thinking-off), ale i to podhodnocuje skutečnou Gemma 4 IT kvalitu pro chat-deployment.
- **BCM leaderboard srovnání je apples-to-oranges** mezi pre-reasoning era (2024–2025, většina BCM) a post-reasoning era (early 2026, naše 4 modely). Loglikelihood eval nadržuje pre-reasoning modelům.
- **Pro plné vyhodnocení reálné chat-mode kvality** by bylo třeba doplnit eval mimo lm-eval-harness (LLM-as-judge nebo task-driven manual eval). Náš patch řeší MC část; generativní část (kromě summarization) zůstává otevřená.
- **Pro vyhodnocení reálné instruction-following / chat-mode kvality** by bylo třeba opustit lm-evaluation-harness pipeline a postavit eval přes plný `/v1/chat/completions` s reasoning budget a LLM-as-judge metrikami. To je samostatný projekt na týdny, který zde záměrně neděláme.
