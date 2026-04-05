import os
from dotenv import load_dotenv
# from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field
import time
# 1. Load Environment Variables
load_dotenv()

# 2. Configure Global LLM Settings
# Gemini 1.5 Flash is highly recommended for speed and cost-efficiency in hackathons
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(
    model="gemini-3.1-flash-lite-preview",   # ← no "models/" prefix
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)
# Set it as the default LLM for the entire framework
Settings.llm = llm

# --- Your existing Schema ---
class AdjudicationVerdict(BaseModel):
    verdict: str = Field(description="Must be 'Approved', 'Partially Approved', or 'Rejected'")
    reasoning: str = Field(description="Legal explanation for the decision.")
    
    # Financial Breakdown
    claimed_amount: float = Field(description="The original amount from the bill.")
    approved_amount: float = Field(description="The final amount the insurance will pay.")
    patient_responsibility: float = Field(description="The amount the patient must pay (Co-pay/Deductible).")
    calculation_details: str = Field(description="Step-by-step math (e.g., '10% co-pay applied to $500').")
    
    # Citation
    citation_page: int = Field(description="The exact page number.")
    citation_text: str = Field(description="The specific quote supporting the calculation.")
    source_node_index: int = Field(description="Index of the source block used.")


# --- Updated Adjudicator Function ---
def adjudicate_line_item(item, evidence_nodes):
    context_blocks = []
    for i, n in enumerate(evidence_nodes):
        block = f"--- SOURCE BLOCK [{i}] (Page {n.metadata['page_number']}) ---\n{n.text}"
        context_blocks.append(block)
    
    context_text = "\n\n".join(context_blocks)

    prompt_template = """
    You are a Senior Insurance Claims Adjuster. 
    Perform a financial reconciliation of the Bill Item against the Policy Context.

    ### Hospital Bill Item:
    Description: {description}
    Claimed Amount: {amount}

    ### Provided Policy Context:
    {context}

    ### Instructions:
    1. **Identify Coverage:** Is this procedure covered?
    2. **Find Financial Rules:** Look for keywords like 'Co-pay', 'Coinsurance', 'Deductible', 'Limit', or 'Cap'.
    3. **Calculate:** - If the policy covers only a percentage (e.g., 80%), calculate that percentage of the claimed amount.
       - If there is a fixed Co-pay (e.g., $50), subtract it.
       - If the amount exceeds a 'Sub-limit' (e.g., Max $200), the approved amount is the limit.
    4. **Verdict:** - 'Approved' if 100% is covered.
       - 'Partially Approved' if >0% but <100% is covered.
       - 'Rejected' if 0% is covered.
    
    Return the result in JSON.
    """

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=AdjudicationVerdict,
        prompt_template_str=prompt_template,
        llm=llm
    )
    retries=3
    for attempt in range(retries):
        try:
            verdict = program(
                description=item['description'],
                amount=item['amount'],
                context=context_text
            )
            break
        except Exception as e:
            if attempt < retries - 1:
                print(f"⚠️ LLM attempt {attempt+1} failed: {e} — retrying in 5s...")
                time.sleep(5)
            else:
                raise e

    # Link back to coordinates for highlighting
    try:
        source_node = evidence_nodes[verdict.source_node_index]
        polygon_coords = source_node.metadata.get("polygon")
    except:
        polygon_coords = None

    return {
        "item_description": item['description'],
        "verdict": verdict.verdict,
        "math": {
            "claimed": verdict.claimed_amount,
            "approved": verdict.approved_amount,
            "patient_pays": verdict.patient_responsibility,
            "breakdown": verdict.calculation_details
        },
        "citation": {
            "page": verdict.citation_page,
            "text": verdict.citation_text,
            "polygon": polygon_coords
        },
        "reasoning": verdict.reasoning
    }