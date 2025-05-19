# Climate Data Visualization with Fine-Tuned LLM Function Calling


##  Overview

This project demonstrates how to fine-tune Microsoft's Phi-2 model to act as an intelligent router for climate data visualization. The system:

1. **Understands natural language requests** about temperature trends
2. **Selects appropriate visualization tools** (line plots, scatter plots with trends)
3. **Executes the full pipeline** from data filtering to plot generation

Key features:
- QLoRA fine-tuning for efficient adaptation
- Structured JSON output for reliable tool calling
- End-to-error handling from parsing to execution
- Modular tool registry system

##  Technical Stack

- **Core LLM**: `microsoft/phi-2` (3B parameter model)
- **Fine-Tuning**: QLoRA (4-bit quantization + LoRA adapters)
- **Libraries**:
  - Transformers, PEFT, TRL (Hugging Face ecosystem)
  - Bitsandbytes (4-bit quantization)
  - Pandas, Matplotlib (data processing/visualization)

## Dataset
- **Kaggle** - https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/adivarma93/LLM_temperaturedata.git
   cd LLM_temperaturedata


Example Queries
Natural Language Query	Visualization Type
"Show Paris temperatures 1960-2000 as line plot"	Line plot
"Tokyo temperature trends 1970-2010 with scatter plot"	Scatter + trend line
 License

MIT License - See LICENSE for details.
 Acknowledgements

    Microsoft for Phi-2 model
    Hugging Face for Transformers/PEFT/TRL libraries
    Berkeley for bitsandbytes quantization

    Hugging Face for Transformers/PEFT/TRL libraries

    Berkeley for bitsandbytes quantization
