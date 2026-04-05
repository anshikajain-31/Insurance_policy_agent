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

















