# FAISS

* HNSW (Hierarchical Navigable Small World)
  * Why: It is significantly faster for queries than a Flat index.
  * Trade-off: It results in a larger file size on disk, but on Apple Silicon, the search speed is nearly instantaneous even with hundreds of thousands of documents.


# Embedding model

* currently: all-MiniLM-L6-v2
    * fast and light, but might not have best performance
* alternatives with better accuracy: BGE or Nomic
    * BAAI/bge-small-en-v1.5: Similar in size but generally ranks higher on retrieval benchmarks.
    * nomic-embed-text-v1.5: Better for longer documents, as it supports a much larger context window (up to 8k tokens)


# LLM Model

* baseline: Gemma3-1b
  * https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF
  * variants
     * Q4_K_M - 806 MB - 4-bit



* Liquid AI 1.2 model
  * LiquidAI/LFM2-1.2B-GGUF
    * Q4_K_M - 796 MB - 4 bit

* Liquid AI 1.2 RAG model
  * https://huggingface.co/LiquidAI/LFM2-1.2B-RAG

## Measurements 1 - max_tokens 200

temperature = 0
max_tokens = 200

model | avg LLM generation time
---- | --- 
LFM2 350m | 0.6044 ± 0.0255 seconds (avg of 5 runs), 0.6059 ± 0.0265 seconds (avg of 5 runs)
LFM2 1.2b | 2.1979 ± 0.0362 seconds (avg of 5 runs), 2.2052 ± 0.0198 seconds (avg of 5 runs)
LFM2 1.2b RAG | 1.8304 ± 0.0265 seconds (avg of 5 runs), 1.8365 ± 0.0245 seconds (avg of 5 runs)
Gemma3 1b | 1.6662 ± 0.1678 seconds (avg of 5 runs), 1.7226 ± 0.1724 seconds (avg of 5 runs)


## LFM-350 output
The Sinclair Sovereign was a high-end calculator introduced by Sinclair Radionics in 1976. It cost between GB £30 and GB £60 at the time of its introduction. The cost was significantly lower compared to other calculators of the era, which could be purchased for under GB £5

## LFM-1.2b output
The Sinclair Sovereign was a high-end end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was made with a case of pressed steel and came in various finishes, costing between GB £30 and GB £60 at the time. The chrome plated version in December 1976 cost GB £30, and the gold @-@ plated version cost GB £60, including VAT. However, due to small profit margins, the Sovereign was not a commercial success and Sinclair ended up selling it at a loss.

## LFM-1.2b RAG output
The Sinclair Sovereign was a high-end end calculator introduced by Clive Sinclair's company Sinclair Radionics in 1976. It was an attempt to escape from the unprofitable low-end market and was one of the last calculators Sinclair produced. The cost of the Sinclair Sovereign varied depending on the finish, ranging from GB £30 for a chrome-plated version and GB £60 for a gold-plated version, including VAT. However, due to small profit margins, Sinclair ended up selling the Sovereign at a loss, making it not a commercial success.

## Gemma 3 1b output
The Sinclair Sovereign was a high-end calculator introduced by Clive Sinclair in 1976. It cost between £30 and £60 at the time, but it was sold at a loss and not a commercial success. It was made in England and was designed by John Pemberton, who won a Design Council award. It had a Mostek MK50321N main integrated circuit, a small memory register, a LED display, and could perform basic mathematical operations


## Orignal document
The Sinclair Sovereign was a high @-@ end calculator introduced by Clive Sinclair 's company Sinclair Radionics in 1976 . It was an attempt to escape from the unprofitable low end of the market , and one of the last calculators Sinclair produced . Made with a case of pressed steel that a variety of finishes , it cost between GB £ 30 and GB £ 60 at a time when other calculators could be purchased for under GB £ 5 . A number of factors meant that the Sovereign was not a commercial success , including the cost , high import levies on components , competition from cheaper calculators manufactured abroad , and the development of more power @-@ efficient designs using liquid @-@ crystal displays . Though it came with a five @-@ year guarantee , issues such as short battery life limited its usefulness . The company moved on to producing computers soon afterwards . 
The design by John Pemberton won a Design Council award , and there are examples of the Sovereign in the Museum of Modern Art in New York . It had a Mostek MK50321N main integrated circuit and a small memory register , a LED display , and could perform a variety of a number of basic mathematical operations besides four @-@ function arithmetic . 


## Measurements 1 - max_tokens 50

temperature = 0
max_tokens = 50

model | avg LLM generation time
---- | --- 
LFM2 1.2b | 1.2073 ± 0.0357 seconds (avg of 5 runs)
Gemma3 1b | 0.8827 ± 0.1609 seconds (avg of 5 runs) 



# TODOs

* have claude run experiments
 * make length and temp configurable
 * specify which models
 * output documenbt
* re-index with documen title
* test gemma3 270 on RPI
* construct a few better test cases to test efficiency
* add baseline to results --> without RAG
