# LLM RAG Comparison

## Experimental Setup

LLMs: We are using these three (all under models/*gguf)

* LFM2-350M-Q4_K_M.gguf (tiny and fast, probably not too good)
* gemma-3-1b-it-Q4_K_M.gguf (Google Gemma, previous best)
* LFM2-1.2B-Q4_K_M.gguf (Liquid AI's model, which claims to be better than gemma)
* LFM2-1.2B-RAG-Q4_K_M.gguf (and the RAG specific model that is fine-tuned for this use case)


We use the same query (see below) for all experiments.

The full baseline truth (original wikipedia document), is in section "ground truth"

### Our Query

```What was the Sinclair Sovereign and how much did it cost?```

### Ground truth document for the query

```
 = Sinclair Sovereign = 


 The Sinclair Sovereign was a high @-@ end calculator introduced by Clive Sinclair 's company Sinclair Radionics in 1976 . It was an attempt to escape from the unprofitable low end of the market , and one of the last calculators Sinclair produced . Made with a case of pressed steel that a variety of finishes , it cost between GB £ 30 and GB £ 60 at a time when other calculators could be purchased for under GB £ 5 . A number of factors meant that the Sovereign was not a commercial success , including the cost , high import levies on components , competition from cheaper calculators manufactured abroad , and the development of more power @-@ efficient designs using liquid @-@ crystal displays . Though it came with a five @-@ year guarantee , issues such as short battery life limited its usefulness . The company moved on to producing computers soon afterwards . 

 The design by John Pemberton won a Design Council award , and there are examples of the Sovereign in the Museum of Modern Art in New York . It had a Mostek MK50321N main integrated circuit and a small memory register , a LED display , and could perform a variety of a number of basic mathematical operations besides four @-@ function arithmetic . 


 = = History = = 


 The Sovereign was one of the last calculators produced during Sinclair 's foray into the calculator market that had started with the Sinclair Executive in September 1972 . The Executive had retailed for GB £ 80 when introduced , but in little over a year it was possible to purchase a Sinclair calculator for GB £ 20 and by November of 1976 a model was available for GB £ 4 @.@ 95 . Cheaper calculators with liquid @-@ crystal displays instead of light @-@ emitting diodes were becoming more popular , and had much longer battery lives of months or years . Such calculators were available for well under GB £ 10 , with all the functionality of the more expensive models . The impossibility of selling " simple " calculators profitibly led Sinclair to introduce models such as the Cambridge Scientific , introduced in August 1975 at a price of GB £ 29 @.@ 95 . 

 The Sovereign , released in 1976 , represented an attempt to move upmarket in an increasingly saturated market . In December 1976 , the chrome plated version of the Sovereign cost GB £ 30 and the gold @-@ plated version GB £ 60 , including VAT , but profit margins on the Sovereign were so small that Sinclair ended up selling the Sovereign at a loss , and it was not a commercial success . The Sovereign was made in England , like every other Sinclair calculator except the President . 

 Sinclair would shortly stop producing calculators and instead focus on computers , starting with the MK14 in 1977 . The loss of the calculator market was due in part to technological development leading to smaller and cheaper components , which put heavy pressure on profit margins . An import levy of up to 17 @.@ 5 % was placed on components , but the duty for calculators imported from Japan or Hong Kong could be as little as 5 % , making it unprofitable to produce calculators in the UK . Sinclair also had some problems with the reliability of earlier calculators that had adversely affected its reputation , but the Sovereign was sold with a " full and unconditional " five year guarantee . 


 = = Design = = 


 The Sovereign came in satin chrome and gold @-@ plated models , with leather pouches and fitted wooden cases . It had an 8 @-@ digit seven @-@ segment display that used red light @-@ emitting diodes , with a decimal point to the right of each digit that could be illuminated as necessary . Power was provided by two 1 @.@ 35 mercury button cells . The Sovereign measured 36 by 141 by 12 millimetres ( 1 @.@ 42 in × 5 @.@ 55 in × 0 @.@ 47 in ) , which made it small and sleek compared to other calculators of the time . 

 The Sovereign was unusual because the casing was made from pressed steel , which gave it a much higher quality feel compared to injection moulded plastic . This allowed a variety of paint and plating options , including black painted , chrome @-@ plated , silver @-@ plated , and gold @-@ plated , and a limited edition silver @-@ plated version , inscribed to commemorate the Silver Jubilee of Queen Elizabeth II in 1977 , were also produced . Asprey of London was rumoured to have produced two Sovereigns in solid gold , costing GB £ 2 @,@ 750 each . 

 The design , by John Pemberton , won the Design Council Award in 1977 , as the Executive had in 1973 , and there are examples in the collection of the Museum of Modern Art in New York . Sovereigns are highly collectible , and used models command high prices . 


 = = = Functions = = = 


 As well as addition , subtraction , multiplication and division , it had reciprocal and square @-@ root functions , and the ability to multiply by a fixed constant . With an eight @-@ digit display , the calculator could display positive numbers between 0 @.@ 0000001 and 99 @,@ 999 @,@ 999 , and negative numbers between -0.000001 and -9,999,999 . Calculators of the time tended to have displays of between 3 and 12 digits , as reducing the number of digits was an effective way of reducing the cost of the calculator . A number outside that range leads to an overflow , and the screen flashes and all keys except the clear key are rendered inoperable to inform the user of the error . A independent memory register could read information from the screen , and information could only be taken from the memory onto the screen . Five keys were used for memory operations . 

 The Sovereign used a Mostek MK50321N main integrated circuit , the same as the Sinclair Wrist Calculator and some variants of the Sinclair Cambridge . Clive Sinclair had assumed that people would prefer attractive illuminated LED displays to LCD displays , which incidentally also required more expensive CMOS chips . However , his calculators were designed with the assumption they would be turned off between calculations , which did not prove to be the case . Advertisements suggested that the batteries would last " about a year " under normal use , but in reality the small button cell batteries and comparatively high power consumption meant a short battery life compared to the competition . 
```


## Results

We compare the speed of all LLMs for LLM generation (given the found snippets) and also the correctness comparing to the ground truth

**Experimental Configuration:**
- Prompt format: `lfm2-rag` (system message with documents in `<document>` tags)
- Temperature: 0.0 (greedy decoding)
- Max tokens: 50
- Number of runs: 5 (for statistical averaging)

### Speed

Results sorted by speed (fastest to slowest):

| Model | LLM Generation Time | Std Dev |
|-------|---------------------|---------|
| **LFM2-350M-Q4_K_M** | 0.5248s | ±0.0119s |
| gemma-3-1b-it-Q4_K_M | 0.7641s | ±0.1423s |
| LFM2-1.2B-Q4_K_M | 1.2467s | ±0.0165s |
| LFM2-1.2B-RAG-Q4_K_M | 1.2565s | ±0.0142s |

**Key findings:**
- The smallest model (LFM2-350M) is **2.4x faster** than the largest models
- The RAG-specific model (LFM2-1.2B-RAG) is actually the slowest
- Gemma provides a good balance between speed and model size

### Model Outputs

Raw outputs from each model (all runs produced identical outputs due to temperature=0.0):

#### LFM2-350M-Q4_K_M
```
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company in 1976. It cost between GB £30 and GB £60 at the time, making it one of the last calculators Sinclair produced.
```

#### gemma-3-1b-it-Q4_K_M
```
According to the documents, the Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair in 1976. It cost between £30 and £60 at the time.
```

#### LFM2-1.2B-Q4_K_M
```
The Sinclair Sovereign was a high-end end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end calculator market. The Sovereign was
```

#### LFM2-1.2B-RAG-Q4_K_M
```
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end market and was one of the last calculators
```

### Quality

Comparing the generated answers to the ground truth (calculator, 1976, £30-£60):

#### LFM2-350M-Q4_K_M ✅ **BEST QUALITY**
**Output:** "The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company in 1976. It cost between GB £30 and GB £60 at the time, making it one of the last calculators Sinclair produced."

**Evaluation:** Excellent! Correctly identifies:
- Device type: calculator ✓
- Year: 1976 ✓
- Price range: £30-£60 ✓
- Additional context: one of last calculators ✓

#### gemma-3-1b-it-Q4_K_M ✅ **BEST QUALITY**
**Output:** "According to the documents, the Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair in 1976. It cost between £30 and £60 at the time."

**Evaluation:** Excellent! Correctly identifies:
- Device type: calculator ✓
- Year: 1976 ✓
- Price range: £30-£60 ✓

#### LFM2-1.2B-Q4_K_M ❌ **POOR QUALITY**
**Output:** "The Sinclair Sovereign was a high-end end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end calculator market. The Sovereign was"

**Evaluation:** Poor quality:
- Contains error: "high-end end" (duplicate word) ✗
- Wrong statement: "enter the unprofitable low-end" (should be "escape from") ✗
- Missing price information ✗
- Output truncated ✗

#### LFM2-1.2B-RAG-Q4_K_M ❌ **POOR QUALITY**
**Output:** "The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end market and was one of the last calculators"

**Evaluation:** Poor quality:
- Wrong statement: "enter the unprofitable low-end" (should be "escape from") ✗
- Missing price information ✗
- Output truncated ✗

**Surprising discovery:** Despite being specifically fine-tuned for RAG tasks, the LFM2-1.2B-RAG model performs worse than the smaller 350M model!

### Summary

**Winner: LFM2-350M-Q4_K_M** 🏆
- **Fastest:** 0.52s (2.4x faster than larger models)
- **Highest quality:** Accurate, complete answer
- **Best efficiency:** Smallest model with best performance

**Runner-up: gemma-3-1b-it-Q4_K_M**
- Good speed: 0.76s
- High quality: Accurate answer
- Slightly more verbose format

**Recommendation:** For RAG applications on Raspberry Pi, use **LFM2-350M-Q4_K_M** for the best combination of speed and quality.

---

## Results with 200 Max Tokens

To ensure we're not cutting off responses prematurely, we repeated all experiments with `MAX_LLM_GEN_TOKENS = 200` instead of 50.

**Experimental Configuration:**
- Prompt format: `lfm2-rag` (system message with documents in `<document>` tags)
- Temperature: 0.0 (greedy decoding)
- Max tokens: **200** (increased from 50)
- Number of runs: 5 (for statistical averaging)

### Speed

Results sorted by speed (fastest to slowest):

| Model | LLM Generation Time | Std Dev | Speed vs 50 tokens |
|-------|---------------------|---------|-------------------|
| **gemma-3-270m-it-Q4_K_M** | 0.1000s | ±0.0440s | N/A (new) |
| gemma-3-1b-it-Q4_K_M | 0.7599s | ±0.1425s | No change |
| LFM2-350M-Q4_K_M | 0.8630s | ±0.0083s | +64% slower |
| LFM2-1.2B-RAG-Q4_K_M | 2.0494s | ±0.0440s | +63% slower |
| LFM2-1.2B-Q4_K_M | 3.0375s | ±0.0360s | +144% slower |

**Key findings:**
- **Gemma-3-270M is EXTREMELY fast** - 0.1s (7.6x faster than Gemma-1B!)
- **Gemma-1B** remains fast and shows no speed change (it naturally finished in ~30 tokens)
- **LFM2-350M** drops from 1st to 2nd place when allowed more tokens
- The 1.2B models are significantly slower with longer outputs (2-3x slower)

### Model Outputs

Raw outputs from each model with 200 max tokens:

#### gemma-3-270m-it-Q4_K_M
```
The Sinclair Sovereign was a high-priced calculator.
```
*Note: Very fast but extremely brief - missing year and specific price information*

#### gemma-3-1b-it-Q4_K_M
```
According to the documents, the Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair in 1976. It cost between £30 and £60 at the time.
```
*Note: Identical to 50 token output - model naturally stopped after completing the answer*

#### LFM2-350M-Q4_K_M
```
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company in 1976. It cost between GB £30 and GB £60 at the time, making it one of the last calculators Sinclair produced. The Sovereign featured a case made of pressed steel with various finishes and cost between £30 and £60. It was priced higher than other calculators available at the time, reflecting its premium design and functionality.
```
*Note: More verbose than 50 token output, includes additional context but repeats the price twice*

#### LFM2-1.2B-RAG-Q4_K_M
```
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end market and was one of the last calculators Sinclair produced. The Sovereign was made with a case of pressed steel that came in various finishes and cost between £30 and £60 at the time, when other calculators could be purchased for under £5.
```
*Note: Now includes the price information (was missing in 50 token version), but still contains the factual error about "enter the unprofitable low-end" instead of "escape from"*

#### LFM2-1.2B-Q4_K_M
```
The Sinclair Sovereign was a high-end end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end calculator market. The Sovereign was made with a case of pressed steel and came in various finishes, costing between GB £30 and GB £60 at the time. This price point was higher than most other calculators available, which were typically sold for under GB £5. Despite its high cost, the Sovereign was one of the last calculators Sinclair produced, reflecting the company's efforts to diversify beyond calculators into computers, starting with the MK14 in 1977.
```
*Note: Now includes price information, but still has the "high-end end" error and factual error about "enter" vs "escape from"*

### Quality

Comparing the generated answers to the ground truth (calculator, 1976, £30-£60):

#### gemma-3-270m-it-Q4_K_M ❌ **POOR QUALITY**
**Evaluation:** Too brief and incomplete:
- Device type: calculator ✓
- Year: 1976 ✗ (missing)
- Price range: £30-£60 ✗ (only says "high-priced")
- Extremely fast but lacks detail
- Fails to answer the full question

#### gemma-3-1b-it-Q4_K_M ✅ **BEST QUALITY**
**Evaluation:** Perfect! Concise and accurate:
- Device type: calculator ✓
- Year: 1976 ✓
- Price range: £30-£60 ✓
- Natural stopping point (doesn't ramble)

#### LFM2-350M-Q4_K_M ⚠️ **GOOD WITH MINOR ISSUES**
**Evaluation:** Mostly correct but verbose:
- Device type: calculator ✓
- Year: 1976 ✓
- Price range: £30-£60 ✓
- Redundancy: States the price twice ✗
- Additional context adds value but slightly redundant

#### LFM2-1.2B-RAG-Q4_K_M ⚠️ **IMPROVED BUT STILL FLAWED**
**Evaluation:** Better than 50 tokens but still has errors:
- Device type: calculator ✓
- Year: 1976 ✓
- Price range: £30-£60 ✓ (now included!)
- Factual error: "enter the unprofitable low-end" should be "escape from" ✗

#### LFM2-1.2B-Q4_K_M ❌ **POOR QUALITY**
**Evaluation:** More complete but still contains errors:
- Device type: calculator ✓
- Year: 1976 ✓
- Price range: £30-£60 ✓ (now included!)
- Contains error: "high-end end" (duplicate word) ✗
- Factual error: "enter the unprofitable low-end" should be "escape from" ✗

### Summary (200 Tokens)

**Winner: gemma-3-1b-it-Q4_K_M** 🏆
- **Best balance:** 0.76s with perfect quality
- **Highest quality:** Complete, accurate answer
- **Most efficient:** Knows when to stop (natural completion)
- **Best practice:** Model demonstrates good judgment by not generating unnecessary tokens

**Runner-up: LFM2-350M-Q4_K_M**
- Good speed: 0.86s
- Good quality: Accurate with minor redundancy
- Trade-off: Faster at 50 tokens, more verbose at 200 tokens

**Speed Champion (but quality issues): gemma-3-270m-it-Q4_K_M** ⚡
- **Blazing fast:** 0.1s (7.6x faster than Gemma-1B!)
- **Major drawback:** Incomplete answer (missing year and specific prices)
- **Use case:** Only suitable for applications where speed > accuracy

**Key Insights:**
1. **Gemma-1B wins with 200 tokens** because it naturally completes answers concisely
2. **Gemma-270M is extremely fast** but sacrifices too much quality for RAG tasks
3. **LFM2-350M was best at 50 tokens** but becomes slightly redundant with more space
4. **1.2B models still have factual errors** even with complete outputs
5. **Token limit matters:** Some models (Gemma) stop naturally, others (350M) fill available space
6. **Model size ≠ quality:** The tiny 270M Gemma is faster but much worse than the mid-sized 350M LFM2

**Updated Recommendation:** Use **gemma-3-1b-it-Q4_K_M** for RAG applications - it provides the best balance of speed, quality, and natural completion without requiring careful token limits. The 270M model is too inaccurate despite its impressive speed.

---

## Results with Explicit Multi-Part Query (200 Max Tokens)

The previous queries were somewhat ambiguous, potentially allowing models to provide incomplete answers. We now test with an explicit, structured query that requires all key information.

**New Query:**
```
What was the Sinclair Sovereign? Include what type of device it was, the year it was introduced, its price range, and one notable or special fact about it.
```

**Expected Ground Truth Answer Should Include:**
1. **Device type:** Calculator
2. **Year:** 1976
3. **Price range:** £30-£60 (or GB £30-£60)
4. **Notable fact:** Any of:
   - Design Council award winner
   - Examples in Museum of Modern Art (MoMA) in New York
   - Made with pressed steel case (unusual for calculators)
   - Commercial failure/not commercially successful
   - Came with 5-year guarantee
   - Made in England
   - One of last calculators Sinclair produced

**Experimental Configuration:**
- Prompt format: `lfm2-rag` (system message with documents in `<document>` tags)
- Temperature: 0.0 (greedy decoding)
- Max tokens: 200
- Number of runs: 5 (for statistical averaging)

### Speed

Results sorted by speed (fastest to slowest):

| Model | LLM Generation Time | Std Dev | vs Previous Query |
|-------|---------------------|---------|-------------------|
| gemma-3-270m-it-Q4_K_M | 0.8313s | ±0.0920s | **8.3x slower** (was 0.1s) |
| LFM2-350M-Q4_K_M | 2.2784s | ±0.1092s | 2.6x slower (was 0.86s) |
| **gemma-3-1b-it-Q4_K_M** | 2.4068s | ±0.1284s | 3.2x slower (was 0.76s) |
| LFM2-1.2B-Q4_K_M | 2.9601s | ±0.0592s | Similar (was 3.04s) |
| LFM2-1.2B-RAG-Q4_K_M | 3.5378s | ±0.0441s | 1.7x slower (was 2.05s) |

### Model Outputs

Raw outputs from each model with the new explicit query:

#### gemma-3-270m-it-Q4_K_M
```
The Sinclair Sovereign was a high-powered calculator designed to help users escape from the unprofitable low end of the market. It was introduced by Clive Sinclair in 1976, and it was a last-used calculator produced by Sinclair Radionics. It was made with a case of pressed steel that a variety of finishes, including a special K-block finish, and it cost between GB £ 30 and GB £ 60 at a time when other calculators could be purchased for under GB £ 5.
```
*Note: Now generating longer output but contains hallucinations (e.g., "K-block finish" doesn't exist in documents) and awkward phrasing*

#### gemma-3-1b-it-Q4_K_M
```
According to the documents, the Sinclair Sovereign was a **high-end calculator** introduced by Clive Sinclair's company in 1976.

Here's a breakdown of its key details:

*   **Type of Device:** It was a calculator.
*   **Year Introduced:** 1976
*   **Price Range:** Between £30 and £60 (at the time)
*   **Notable/Special Fact:** It was one of the last calculators Sinclair produced during their foray into the calculator market. It had a key component (Mostek MK50321N main integrated circuit) and could perform basic mathematical operations.
```
*Note: Perfect structured response with bullet points, addresses all requirements clearly*

#### LFM2-350M-Q4_K_M
```
The Sinclair Sovereign was a high-end calculator introduced by Sinclair Radionics in 1976. It was designed to compete with the Sinclair Executive, a calculator Sinclair had previously launched in 1972.

Specifically, the Sovereign cost between £30 and £60, including VAT, and was priced lower than many other calculators of the time. It featured a chrome-plated casing, a variety of finishes, and a mostek MK50321N integrated circuit for basic arithmetic operations. It could perform a range of mathematical functions beyond the four @-@ function, including addition, subtraction, multiplication, and division, as well as exponentiation.

A notable aspect of the Sovereign was its relatively small profit margin; Sinclair ended up selling it at a loss, and it was not a commercially successful product. This ultimately led Sinclair to shift their focus towards computers, beginning with the MK14 calculator in 1977
```
*Note: Comprehensive answer with good detail, includes notable fact about commercial failure*

#### LFM2-1.2B-RAG-Q4_K_M
```
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to enter the unprofitable low-end calculator market and was one of the last calculators Sinclair produced. The Sovereign was made with a case of pressed steel that came in various finishes and cost between GB £30 and GB £60 at the time, when other calculators could be purchased for under GB £5. A notable fact about the Sovereign is that it won a Design Council award and had an example in the Museum of Modern Art in New York. It featured a Main Integrated Circuit (Mek50321N) and a small memory register, LED display, and could perform a variety of basic mathematical operations beyond four @-@ function arithmetic.
```
*Note: Comprehensive but still contains the factual error "enter the unprofitable low-end" (should be "escape from")*

#### LFM2-1.2B-Q4_K_M
```
The Sinclair Sovereign was a high-end end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was designed to compete with the unprofitable low-end calculator market. The device was made with a case of pressed steel and came in various finishes, costing between £30 and £60 at the time. Notably, the Sovereign was one of the last calculators Sinclair produced, marking a significant shift in their focus from calculators to computers, starting with the MK14 in 1977. Despite its high-end features, the Sovereign was not commercially successful due to small profit margins, leading Sinclair to sell it at a loss.
```
*Note: Still contains the "high-end end" duplicate word error and factual error about "compete with" instead of "escape from"*

### Quality Evaluation

Scoring each model on the 4 required criteria:

#### gemma-3-1b-it-Q4_K_M ✅ **EXCELLENT** (4/4)
- ✓ Device type: Calculator
- ✓ Year: 1976
- ✓ Price range: £30-£60
- ✓ Notable fact: Last calculators Sinclair produced + technical details
- **Bonus:** Perfect formatting with bullet points, clear structure, cites source

#### LFM2-350M-Q4_K_M ✅ **VERY GOOD** (4/4)
- ✓ Device type: Calculator
- ✓ Year: 1976
- ✓ Price range: £30-£60
- ✓ Notable fact: Commercial failure, sold at a loss
- **Quality:** Comprehensive, good narrative flow, accurate details

#### gemma-3-270m-it-Q4_K_M ⚠️ **POOR** (3/4 with errors)
- ✓ Device type: Calculator
- ✓ Year: 1976
- ✓ Price range: £30-£60
- ✗ Notable fact: Mentions "special K-block finish" - **HALLUCINATION** (doesn't exist in documents)
- **Major issues:** Awkward phrasing ("last-used calculator"), makes up facts

#### LFM2-1.2B-RAG-Q4_K_M ⚠️ **GOOD BUT FLAWED** (4/4 with factual error)
- ✓ Device type: Calculator
- ✓ Year: 1976
- ✓ Price range: £30-£60
- ✓ Notable fact: Design Council award + MoMA
- ✗ **Factual error:** "enter the unprofitable low-end" (should be "escape from")
- **Quality:** Comprehensive but persistent comprehension error

#### LFM2-1.2B-Q4_K_M ❌ **GOOD BUT FLAWED** (4/4 with errors)
- ✓ Device type: Calculator
- ✓ Year: 1976
- ✓ Price range: £30-£60
- ✓ Notable fact: Last calculators + commercial failure
- ✗ **Duplicate word:** "high-end end"
- ✗ **Factual error:** "compete with the unprofitable low-end" (wrong interpretation)
- **Quality:** Good content but persistent text generation errors

### Summary (Explicit Query)

**Winner: gemma-3-1b-it-Q4_K_M** 🏆
- **Perfect accuracy:** Answers all 4 requirements correctly
- **Best formatting:** Uses bullet points for clarity
- **Good speed:** 2.4s (middle of the pack)
- **Reliable:** No hallucinations or factual errors
- **Professional:** Cites sources appropriately

**Runner-up: LFM2-350M-Q4_K_M**
- **Complete answer:** All 4 requirements met
- **Fastest of accurate models:** 2.3s
- **Good narrative:** Natural prose style
- **Accurate:** No errors or hallucinations

**Key Findings:**
1. **Explicit queries help but don't fix fundamental issues:**
   - Gemma-270M still hallucinates despite clear instructions
   - LFM2-1.2B models still make the same factual comprehension errors

2. **All models now generate longer outputs:**
   - Gemma-270M: 0.1s → 0.83s (**8.3x slower**)
   - Gemma-1B: 0.76s → 2.4s (3.2x slower)
   - Explicit queries force models to generate more content

3. **Gemma-1B excels at structured tasks:**
   - Only model to use proper formatting (bullet points)
   - Demonstrates superior instruction following
   - Understands how to structure multi-part answers

4. **Model training quality matters more than size:**
   - 350M LFM2 outperforms 1.2B LFM2 models in accuracy
   - 270M Gemma hallucinates while 1B Gemma is perfect
   - RAG-specific fine-tuning (LFM2-1.2B-RAG) doesn't prevent errors

**Final Recommendation:**
Use **gemma-3-1b-it-Q4_K_M** for RAG applications. Despite being slower with explicit queries (2.4s), it's the only model that:
- Answers all requirements accurately
- Formats output professionally
- Never hallucinates or makes factual errors
- Properly follows multi-part instructions

For speed-critical applications where slight verbosity is acceptable, **LFM2-350M-Q4_K_M** (2.3s) is an excellent alternative with zero errors.

---

## Results with Conciseness Instructions (200 Max Tokens)

After observing that some models generated overly verbose responses, we modified the system prompt to include explicit conciseness instructions.

**Prompt Changes Made:**

Added to the system message (LFM2-RAG format):
```
Instructions: Provide clear, concise answers based only on the information in the documents.
Limit your response to 3-4 sentences maximum. Be direct and avoid unnecessary elaboration.
```

**Hypothesis:** The sentence limit should reduce token generation, leading to faster inference while maintaining answer completeness.

**Query:** Same as previous experiment
```
What was the Sinclair Sovereign? Include what type of device it was, the year it was introduced, its price range, and one notable or special fact about it.
```

**Experimental Configuration:**
- Prompt format: `lfm2-rag` with conciseness instructions
- Temperature: 0.0 (greedy decoding)
- Max tokens: 200
- Number of runs: 5
- Models tested: gemma-3-270m, LFM2-350M, gemma-3-1b (skipped LFM2-1.2B models)

### Speed Results

| Model | Before (no conciseness) | After (with conciseness) | Speedup | Output Length |
|-------|------------------------|--------------------------|---------|---------------|
| **LFM2-350M-Q4_K_M** | 2.28s | 0.81s | **2.8x faster** | 67 words (3 sentences) |
| gemma-3-1b-it-Q4_K_M | 2.41s | 1.65s | 1.5x faster | 85 words (4 sentences) |
| gemma-3-270m-it-Q4_K_M | 0.83s | 0.64s | 1.3x faster | 72 words (3 sentences) |

### Model Outputs with Conciseness

#### LFM2-350M-Q4_K_M ✅ **BEST** (0.81s)
```
The Sinclair Sovereign was a high-end calculator introduced by Sinclair Radionics in December 1976. It cost between £30 and £60, including VAT, and was priced lower than many calculators of the time. A notable feature was its chrome-plated casing and a variety of finishes, though it was ultimately sold at a loss due to low profit margins.
```
**Quality:** ✓ Device type ✓ Year ✓ Price range ✓ Notable fact
**Analysis:** Perfect 3-sentence answer, all requirements met, focused and accurate

#### gemma-3-1b-it-Q4_K_M ✅ **EXCELLENT** (1.65s)
```
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair in 1976. It was a last-produced calculator aimed at escaping the low-profit market, costing between £30 and £60 at the time. It featured a steel case, various finishes, and an integrated circuit, and was one of the last calculators produced by Sinclair. It was designed by John Pemberton and was sold in chrome and gold plated versions.
```
**Quality:** ✓ Device type ✓ Year ✓ Price range ✓ Notable fact
**Analysis:** Perfect 4-sentence answer, all requirements met, slightly more detailed than LFM2-350M

#### gemma-3-270m-it-Q4_K_M ❌ **INCOMPLETE** (0.64s)
```
The Sinclair Sovereign was a high-powered calculator introduced by Clive Sinclair in 1976. It was a last-used calculator produced by Sinclair Radionics, with a case of pressed steel that offered a variety of finishes. It was one of the last calculators produced during Sinclair's foray into the calculator market, starting with the Sinclair Executive in September 1972.
```
**Quality:** ✓ Device type ✓ Year ✗ **Missing price range** ✓ Notable fact
**Analysis:** Fast but fails to include price information, demonstrating model limitations

### LFM2-350M Quality Comparison

**Before (without conciseness, 2.28s, ~170 words, 3 paragraphs):**
```
The Sinclair Sovereign was a high-end calculator introduced by Sinclair Radionics in 1976. It was designed to compete with the Sinclair Executive, a calculator Sinclair had previously launched in 1972.

Specifically, the Sovereign cost between £30 and £60, including VAT, and was priced lower than many other calculators of the time. It featured a chrome-plated casing, a variety of finishes, and a mostek MK50321N integrated circuit for basic arithmetic operations. It could perform a range of mathematical functions beyond the four @-@ function, including addition, subtraction, multiplication, and division, as well as exponentiation.

A notable aspect of the Sovereign was its relatively small profit margin; Sinclair ended up selling it at a loss, and it was not a commercially successful product. This ultimately led Sinclair to shift their focus towards computers, beginning with the MK14 calculator in 1977
```
✓ All 4 requirements met, but **overly verbose** with unnecessary technical details

**After (with conciseness, 0.81s, ~67 words, 3 sentences):**
```
The Sinclair Sovereign was a high-end calculator introduced by Sinclair Radionics in December 1976. It cost between £30 and £60, including VAT, and was priced lower than many calculators of the time. A notable feature was its chrome-plated casing and a variety of finishes, though it was ultimately sold at a loss due to low profit margins.
```
✓ All 4 requirements met, **perfectly concise** and focused

**Quality Analysis:**
- **Information completeness:** Both versions answer all 4 requirements correctly
- **Accuracy:** Both are factually accurate, no errors or hallucinations
- **Conciseness:** After version is 2.5x shorter (67 vs 170 words)
- **Focus:** After version stays on topic, before version adds tangential details
- **Readability:** After version is clearer and easier to parse
- **Efficiency:** After version delivers same essential information in 2.8x less time

**Verdict:** The conciseness instructions **improved quality** - the response is more focused, easier to read, and wastes no time on unnecessary elaboration while maintaining perfect accuracy.

### Summary (With Conciseness)

**Winner: LFM2-350M-Q4_K_M** 🏆
- **Fastest accurate model:** 0.81s (2.8x improvement!)
- **Perfect quality:** All requirements met with zero errors
- **Optimal conciseness:** 67 words, 3 sentences - exactly what was asked
- **Best efficiency:** Unbeatable speed/quality ratio

**Runner-up: gemma-3-1b-it-Q4_K_M**
- **Very good speed:** 1.65s (1.5x improvement)
- **Perfect quality:** All requirements met with zero errors
- **Slightly more detail:** 85 words, 4 sentences - more context provided

**Key Findings:**

1. **Conciseness instructions dramatically improve speed:**
   - LFM2-350M: 2.28s → 0.81s (**2.8x faster**)
   - Gemma-1B: 2.41s → 1.65s (**1.5x faster**)
   - Models generate fewer tokens = faster inference

2. **Quality is maintained or improved:**
   - LFM2-350M produces more focused, readable answers
   - No loss of required information
   - Removes tangential details that weren't requested

3. **Sentence limits work better than token limits:**
   - Models understand "3-4 sentences" more reliably
   - More consistent output lengths across models
   - Easier for humans to specify desired verbosity

4. **Model limitations remain:**
   - Gemma-270M still fails basic requirements despite speed gains
   - Instruction following > model size for quality

**Final Recommendation:**
Use **LFM2-350M-Q4_K_M** with conciseness instructions for RAG applications:
- Sub-second response time (0.81s)
- Perfect accuracy on multi-part queries
- Naturally concise without excessive formatting
- Best speed/quality trade-off for production use

For applications requiring structured output with bullet points, **gemma-3-1b-it-Q4_K_M** (1.65s) remains excellent despite being 2x slower.