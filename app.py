import streamlit as st
import pandas as pd
import traceback ,sys
from ocr import extract_bill_data, extract_policy_metadata
from core.retriver import build_hybrid_index, run_per_item_retrieval
from core.adjuvicator import adjudicate_line_item
import base64
from utils import highlight_clause_in_pdf

st.set_page_config(page_title="Insurance Claim Adjudicator")

st.title("🧾 Insurance Claim Adjudication System")

# -------------------------------
# STEP 1: Upload Policy Document
# -------------------------------
st.header("1. Upload Insurance Policy")

policy_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
if policy_file:
    policy_bytes = policy_file.read()
    policy_file.seek(0)
    st.session_state.policy_bytes = policy_bytes

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
    st.session_state.bm25 = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if policy_file and st.button("Index Policy"):
    try:
        st.write("🔹 Step 1: Extracting metadata...")
        policy_nodes = extract_policy_metadata(policy_file)
        st.write(f"✅ Nodes extracted: {len(policy_nodes)}")
        # policy_nodes = policy_nodes[:50]
        
        st.write("🔹 Step 2: Building index...")
        vector_index, bm25 = build_hybrid_index(policy_nodes)
        st.write("✅ Index built!")

        st.session_state.vector_index = vector_index
        st.session_state.bm25 = bm25
        st.session_state.indexed = True
        st.success("✅ Policy indexed successfully!")

    except Exception as e:
        st.error(f"❌ CRASH: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()
    # -------------------------------
# STEP 2: Upload Bill
# -------------------------------
st.header("2. Upload Hospital Bill")

bill_file = st.file_uploader("Upload Bill PDF", type=["pdf"])

if bill_file and st.session_state.vector_index:
    with st.spinner("Extracting bill data..."):
        bill_data = extract_bill_data(bill_file)

    st.subheader("Bill Summary")
    st.json(bill_data["summary"])

    results = []

    st.subheader("Adjudication Results")

    for i, item in enumerate(bill_data["items"]):
        st.write(f"### Item {i+1}: {item['description']}")

        # Retrieval
        evidence_nodes = run_per_item_retrieval(
            st.session_state.vector_index,
            st.session_state.bm25,
            item
        )

        # Adjudication
        result = adjudicate_line_item(item, evidence_nodes)
        

        results.append({
            "Item": item["description"],
            "Claimed": result["math"]["claimed"],
            "Approved": result["math"]["approved"],
            "Patient Pays": result["math"]["patient_pays"],
            "Verdict": result["verdict"]
        })

        # # Show reasoning
        # with st.expander("📖 View Explanation"):
        #     st.write(result["reasoning"])
        #     st.write("**Calculation:**", result["math"]["breakdown"])
        #     st.write("**Citation:**", result["citation"]["text"])

        # # Highlight button (basic)
        # if st.button(f"Show Clause (Item {i+1})"):
        #     st.info(f"Page: {result['citation']['page']}")
        #     st.code(result["citation"]["polygon"])
        with st.expander(f"📖 Explanation — {result['verdict']}"):
            st.write("**Reasoning:**", result["reasoning"])
            st.write("**Calculation:**", result["math"]["breakdown"])
            st.write("**Citation:**", result["citation"]["text"])
            st.write("**Page:**", result["citation"]["page"])

            if st.button(f"📄 Show Clause in PDF (Item {i+1})"):
               if st.session_state.get("policy_bytes") and result["citation"]["polygon"]:
                  highlighted_pdf = highlight_clause_in_pdf(
                   st.session_state.policy_bytes,
                   result["citation"]["page"],
                   result["citation"]["polygon"]
            )

            # Display PDF with highlight
               b64 = base64.b64encode(highlighted_pdf).decode("utf-8")
               page_num = result["citation"]["page"] - 1  # 0-based for viewer

               st.markdown(f"""
               <iframe
                   src="data:application/pdf;base64,{b64}#page={page_num + 1}"
                   width="100%"
                   height="600px"
                   type="application/pdf"
               ></iframe>
               """, unsafe_allow_html=True)
            else:
               st.info(f"📄 Page {result['citation']['page']}: {result['citation']['text']}")
    # -------------------------------
    # Final Table
    # -------------------------------
    df = pd.DataFrame(results)
    st.dataframe(df)

else:
    st.info("👉 Please upload and index a policy first.")


















# import streamlit as st
# import json
# import os
# import time
# import pandas as pd
# from dotenv import load_dotenv
# from ocr import extract_bill_data, extract_policy_metadata

# # LlamaIndex Gemini Imports
# from llama_index.core import VectorStoreIndex, Settings
# from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# # Load the GOOGLE_API_KEY from .env
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # --- Global Configuration (Free Tier Friendly) ---
# if GOOGLE_API_KEY:
#     # Switched to Gemini 3 Flash to avoid 429 Error on Free Tier
#     Settings.llm = GoogleGenAI(
#         model="gemini-3-flash-preview", 
#         api_key=GOOGLE_API_KEY
#     )
    
#     # Using the standard stable embedding model
#     Settings.embed_model = GoogleGenAIEmbedding(
#         model_name="
# 
# 
# ", 
#         api_key=GOOGLE_API_KEY
#     )
    
#     Settings.chunk_size = 512
# else:
#     st.error("GOOGLE_API_KEY not found in .env file!")

# # --- Page Configuration ---
# st.set_page_config(page_title="ClaimGuard AI", layout="wide")

# st.title("🛡️ Insurance Claim Settlement Agent")
# st.markdown("Automated reconciliation using **Gemini 3 Flash** (Free Tier Optimized).")

# # --- Sidebar: Uploads ---
# with st.sidebar:
#     st.header("1. Upload Documents")
#     bill_file = st.file_uploader("Upload Hospital Bill (PDF/Image)", type=["pdf", "png", "jpg"])
#     policy_file = st.file_uploader("Upload Insurance Policy (PDF)", type=["pdf"])
#     st.divider()
#     st.success("✅ Gemini 3 Flash Pipeline Ready")

# # --- Main Logic ---
# if bill_file and policy_file:
#     if st.button("🚀 Process & Audit Claim"):
#         try:
#             with st.status("Analyzing documents...", expanded=True) as status:
#                 st.write("Step 1: Extracting Bill Data via Azure...")
#                 bill_data = extract_bill_data(bill_file)
                
#                 st.write("Step 2: Mapping Policy Metadata...")
#                 policy_nodes = extract_policy_metadata(policy_file)
                
#                 st.write("Step 3: Building Vector Index...")
#                 index = VectorStoreIndex(policy_nodes)
#                 query_engine = index.as_query_engine(similarity_top_k=5)
                
#                 status.update(label="Analysis Complete!", state="complete", expanded=False)

#             # --- Display Results ---
#             col1, col2 = st.columns([1, 2])

#             with col1:
#                 st.subheader("📋 Bill Summary")
#                 summary = bill_data.get('summary', {})
#                 st.metric("Total Amount", summary.get('total_amount', "N/A"))
#                 st.info(f"**Hospital:** {summary.get('hospital_name', 'N/A')}\n\n**Date:** {summary.get('bill_date', 'N/A')}")

#             with col2:
#                 st.subheader("⚖️ Audit Results")
#                 audit_data = []
                
#                 for item in bill_data.get('items', []):
#                     with st.spinner(f"Auditing: {item['description']}..."):
#                         prompt = f"""
#                         Analyze this claim item: '{item['description']}' for amount {item['amount']}.
#                         Compare it against the provided policy context.
                        
#                         Return ONLY a JSON object:
#                         {{
#                             "status": "Approved" or "Rejected",
#                             "reason": "Short explanation based on policy rules",
#                             "citation": "Page X"
#                         }}
#                         """
#                         response = query_engine.query(prompt)
                        
#                         # Added time.sleep(1) to avoid hitting Free Tier RPM limits
#                         time.sleep(1) 
                        
#                         try:
#                             clean_res = response.response.replace("```json", "").replace("```", "").strip()
#                             res_json = json.loads(clean_res)
#                         except:
#                             res_json = {"status": "Error", "reason": "AI formatting issue", "citation": "N/A"}
                        
#                         audit_data.append({
#                             "Item": item['description'],
#                             "Billed": item['amount'],
#                             "Status": res_json.get('status', 'Unknown'),
#                             "Reason": res_json.get('reason', 'N/A'),
#                             "Citation": res_json.get('citation', 'N/A')
#                         })

#                 # Display table
#                 df = pd.DataFrame(audit_data)
                
#                 def color_status(val):
#                     if val == 'Approved': return 'background-color: #d4edda; color: black'
#                     if val == 'Rejected': return 'background-color: #f8d7da; color: black'
#                     return ''

#                 st.table(df.style.map(color_status, subset=['Status']))

#         except Exception as e:
#             st.error(f"Critical System Error: {e}")
# else:
#     st.info("Upload the hospital bill and policy to start the audit.")































# import streamlit as st
# import os
# import hashlib
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Corrected imports based on your specific ocr.py function names
# from model import aggregate_verdict
# from engine.ocr import extract_bill_data, extract_policy_metadata # Changed from extract_policy_nodes
# from engine.processor import process_and_tag_policy_nodes
# from engine.utils import initialize_storage
# from engine.adjudicator import (
#     run_mandatory_checks, 
#     map_items_to_rules_batch, 
#     filtered_semantic_search, 
#     reason_over_item, 
#     extract_policy_globals,
#     ClaimContext
# )

# # 1. SETUP
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# llm_model = genai.GenerativeModel('gemini-1.5-flash')

# st.set_page_config(page_title="InsurShield AI", layout="wide")
# st.title("🛡️ InsurShield: AI Claim Adjudicator")

# # 2. SESSION STATE
# if "policy_index" not in st.session_state:
#     st.session_state.policy_index = None
# if "vector_index" not in st.session_state:
#     st.session_state.vector_index = None

# # 3. SIDEBAR: POLICY INDEXING
# with st.sidebar:
#     st.header("📋 Policy Setup")
#     policy_pdf = st.file_uploader("Upload Policy Document", type="pdf")
    
#     if policy_pdf and st.button("Index Policy"):
#         with st.spinner("Analyzing Policy..."):
#             # Step 1: OCR (Matches ocr.py function name)
#             raw_nodes = extract_policy_metadata(policy_pdf)
            
#             # Step 2: Tagging (from processor.py)
#             tagged_nodes = process_and_tag_policy_nodes(raw_nodes, llm_model)
            
#             # Step 3: Storage (from utils.py)
#             p_id = hashlib.md5(policy_pdf.getvalue()).hexdigest()[:10]
#             p_idx, v_idx = initialize_storage(tagged_nodes, policy_id=p_id)
            
#             st.session_state.policy_index = p_idx
#             st.session_state.vector_index = v_idx
#             st.success("Policy Indexed and Ready!")

# # 4. MAIN INTERFACE
# if st.session_state.policy_index is None:
#     st.info("Please index a policy in the sidebar to begin.")
# else:
#     col1, col2 = st.columns(2)
#     with col1:
#         st.header("📑 Claim Submission")
#         bill_pdf = st.file_uploader("Upload Hospital Bill", type="pdf")
#         diag = st.text_input("Diagnosis", value="Acute Appendicitis")
#         ped = st.multiselect("Pre-existing Diseases", ["Diabetes", "Hypertension", "Asthma"], default=[])
#         adm_date = st.date_input("Admission Date")
        
#     if bill_pdf and st.button("🚀 Process Claim"):
#         with st.spinner("Adjudicating..."):
#             # Step 4: Extract Bill (from ocr.py)
#             bill_data = extract_bill_data(bill_pdf)
            
#             # Fix: Access total_amount via the 'summary' key defined in ocr.py
#             total_amt_str = bill_data.get('summary', {}).get('total_amount', "0").replace("₹", "").replace(",", "")
#             try:
#                 total_amt = float(total_amt_str)
#             except:
#                 total_amt = 0.0

#             if not bill_data or not bill_data.get('items'):
#                 st.error("Could not read items from bill.")
#                 st.stop()

#             # Create the Pydantic Context
#             claim_ctx = ClaimContext(
#                 admission_date=str(adm_date),
#                 diagnosis_name=diag,
#                 total_claimed_amount=total_amt, # Using the cleaned total
#                 patient_ped_history=ped
#             )

#             # Step 5: Mandatory Checks (from adjudicator.py)
#             failures = run_mandatory_checks(claim_ctx, st.session_state.policy_index, llm_model)
            
#             # Handle Errors
#             if any(f.status == "SYSTEM_ERROR" for f in failures):
#                 st.warning("⚠️ Policy data incomplete. Proceeding with caution.")
            
#             # Step 6: Map Bill Items (from adjudicator.py)
#             mappings = map_items_to_rules_batch(bill_data['items'], llm_model)
            
#             final_item_results = []
#             if mappings:
#                 prog = st.progress(0)
#                 for idx, entry in enumerate(mappings):
#                     # Step 7: Search (from adjudicator.py)
#                     clauses = filtered_semantic_search(
#                         entry['item']['description'], 
#                         entry['rules'], 
#                         st.session_state.vector_index
#                     )
                    
#                     # Step 8: Reason (from adjudicator.py)
#                     # We ensure amount is a float for reasoning
#                     try:
#                         item_amt = float(str(entry['item']['amount']).replace(",", ""))
#                     except:
#                         item_amt = 0.0
                    
#                     # Create temporary dict for reasoner
#                     reasoner_item = {"description": entry['item']['description'], "amount": item_amt}
                    
#                     decision = reason_over_item(reasoner_item, clauses, claim_ctx.model_dump(), llm_model)
                    
#                     final_item_results.append({
#                         "item": entry['item'], 
#                         "decision": decision,
#                         "polygon": entry['item'].get('polygon', []) 
#                     })
#                     prog.progress((idx + 1) / len(mappings))

#                 # Step 9: Aggregate (from model.py)
#                 _, _, _, total_cap, _ = extract_policy_globals(st.session_state.policy_index)
#                 verdict = aggregate_verdict(final_item_results, total_cap, failures)

#                 # --- 5. DISPLAY RESULTS ---
#                 st.balloons()
#                 st.header(f"Final Verdict: {verdict.verdict}")
                
#                 m1, m2, m3 = st.columns(3)
#                 m1.metric("Total Claimed", f"₹{verdict.total_claimed}")
#                 m2.metric("Total Approved", f"₹{verdict.total_approved}")
#                 m3.metric("Deductions", f"₹{verdict.total_deducted}")

#                 st.subheader("Adjudication Citation Report")
#                 display_report = []
#                 for cit in verdict.citation_report:
#                     display_report.append({
#                         "Item": cit.item,
#                         "Approved": f"₹{cit.approved}",
#                         "Reason": cit.reason,
#                         "Location": f"Page {cit.page}, Para {cit.para}"
#                     })
#                 st.table(display_report)
                
#                 if verdict.human_review_required:
#                     st.warning(f"Review required for: {', '.join(verdict.human_review_required)}")