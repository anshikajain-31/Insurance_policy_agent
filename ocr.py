import json
import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from llama_index.core.schema import TextNode

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

def get_file_content(file_input):
    """
    Utility to extract bytes regardless of whether the input is a 
    Streamlit UploadedFile or a local string path.
    """
    if isinstance(file_input, str):
        # Local file path
        with open(file_input, "rb") as f:
            return f.read()
    else:
        # Streamlit UploadedFile object (BytesIO)
        content = file_input.read()
        file_input.seek(0)  # Reset pointer for subsequent reads
        return content

def extract_bill_data(file_input):
    """
    Analyzes a medical bill and returns a structured JSON-compatible dictionary.
    Supports Streamlit uploads and local paths.
    """
    client = DocumentIntelligenceClient(
        endpoint=AZURE_ENDPOINT, 
        credential=AzureKeyCredential(AZURE_KEY)
    )

    # Convert file object/path to raw bytes
    file_content = get_file_content(file_input)

    # 'prebuilt-invoice' is optimized for hospital bills and invoices
    poller = client.begin_analyze_document("prebuilt-invoice", body=file_content)
    result = poller.result()

    if not result.documents:
        return {"error": "No documents detected", "summary": {}, "items": []}

    doc = result.documents[0]
    fields = doc.fields

    def get_val(field_name):
        field = fields.get(field_name)
        return field.content if field else "N/A"

    # 1. Extract Invoice Summary
    bill_summary = {
        "hospital_name": get_val("VendorName").replace("\n", " "),
        "bill_date": get_val("InvoiceDate"),
        "total_amount": get_val("InvoiceTotal")
    }

    # 2. Extract Line Items (The Table)
    line_items = []
    items_field = fields.get("Items")
    
    if items_field and items_field.value_array:
        for item in items_field.value_array:
            item_data = item.value_object
            
            desc_field = item_data.get("Description")
            amt_field  = item_data.get("Amount")
            desc = desc_field.content.strip() if desc_field and desc_field.content else "Unknown"
            amt  = amt_field.content.replace(",", "") if amt_field and amt_field.content else "0.00"
            
            line_items.append({
                "description": desc.strip(),
                "amount": amt.replace(",", "") # Clean for mathematical processing
            })

    return {
        "summary": bill_summary,
        "items": line_items
    }
import pypdf
import io

def extract_policy_metadata(file_input, chunk_size=2):
    """
    Analyzes policy documents and extracts structural metadata (Page numbers & Polygons).
    Splits PDF into chunks to handle Azure free tier 2-page limit.
    """
    client = DocumentIntelligenceClient(
        endpoint=AZURE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_KEY)
    )

    pdf_bytes = get_file_content(file_input)

    # Split into chunks
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    print(f"📄 Total pages: {total_pages}")

    policy_nodes = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)

        # Build chunk PDF
        writer = pypdf.PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        buf = io.BytesIO()
        writer.write(buf)
        chunk_bytes = buf.getvalue()

        try:
            poller = client.begin_analyze_document("prebuilt-layout", body=chunk_bytes)
            result = poller.result()

            if not result.paragraphs:
                print(f"  ⚠️  Pages {start+1}–{end}: no paragraphs found")
                continue

            for para in result.paragraphs:
                region = para.bounding_regions[0] if para.bounding_regions else None
                # Offset local page number (1-based) back to global page number
                local_page = region.page_number if region else 1
                page_num   = start + local_page
                polygon    = region.polygon if region else []
                role       = para.role if para.role else "bodyText"

                policy_nodes.append(
                    TextNode(
                        text=para.content,
                        metadata={
                            "page_number": page_num,
                            "polygon":     json.dumps(polygon),
                            "role":        role
                        }
                    )
                )

            # print(f"  ✅ Pages {start+1}–{end} done | total nodes so far: {len(policy_nodes)}")

        except Exception as e:
            print(f"  ❌ Pages {start+1}–{end} failed: {e} — skipping")
            continue

    print(f"✅ Extraction complete: {len(policy_nodes)} nodes from {total_pages} pages")
    return policy_nodes


# def extract_policy_metadata(file_input):
#     """
#     Analyzes policy documents and extracts structural metadata (Page numbers & Polygons).
#     Supports Streamlit uploads and local paths.
#     """
#     client = DocumentIntelligenceClient(
#         endpoint=AZURE_ENDPOINT, 
#         credential=AzureKeyCredential(AZURE_KEY)
#     )
    
#     # Convert file object/path to raw bytes
#     file_content = get_file_content(file_input)

#     # 'prebuilt-layout' is critical for coordinate mapping and multi-page indexing
#     poller = client.begin_analyze_document("prebuilt-layout", body=file_content)
#     result = poller.result()

#     if not result.paragraphs:
#         return []

#     policy_nodes = []

#     for para in result.paragraphs:
#         region = para.bounding_regions[0] if para.bounding_regions else None
#         page_num = region.page_number if region else 0
#         polygon = region.polygon if region else []
#         role = para.role if para.role else "bodyText"
        
#         policy_nodes.append(
#     TextNode(
#         text=para.content,
#         metadata={
#             "page_number": page_num,
#             "polygon": json.dumps(polygon),
#             "role": role
#         }
#     )
# )
        
#     return policy_nodes