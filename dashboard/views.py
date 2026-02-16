from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from .utils import execute_dashboard_logic
from django.http import JsonResponse, HttpResponse  # Add HttpResponse
import io
import zipfile
import json  # Move this to top instead of inside function
import uuid
import os
from django.conf import settings 
# NOTE: Ensure Django session middleware is enabled in settings.py
# MIDDLEWARE should include: 'django.contrib.sessions.middleware.SessionMiddleware'

def safe_float(value, default=0):
    """Safely convert value to float, handling strings and formatting."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove formatting
        cleaned = value.strip().replace(',', '').replace('$', '').replace('%', '').replace('€', '').replace('£', '')
        # Try converting
        try:
            return float(cleaned)
        except (ValueError, AttributeError):
            return default
    return default

def index(request):
    """
    Renders the frontend HTML.
    """
    return render(request, 'dashboard/index.html')

@csrf_exempt
def process_file(request):
    """
    Handles file upload, sends data to DuckDB, asks AI for SQL, returns JSON.
    Supports filtering via POST parameter 'filters'.
    """
    if request.method == 'POST':
        # Handle single or multiple files
        files = request.FILES.getlist('file') or request.FILES.getlist('files')
        
        if not files:
            return JsonResponse({"error": "No files uploaded"}, status=400)

        try:
            # 1. READ FILES INTO DATAFRAMES
            dfs = {}
            for f in files:
                if f.name.endswith('.xlsx'):
                    # Load all sheets from Excel file
                    sheets = pd.read_excel(f, sheet_name=None)
                    dfs.update(sheets)
                elif f.name.endswith('.csv'):
                    dfs[f.name] = pd.read_csv(f)
            
            if not dfs:
                return JsonResponse({"error": "No valid data found in uploaded files"}, status=400)
            
            # 2. GET FILTERS FROM REQUEST
            # The frontend sends filters as JSON string in POST body
            filters_json = request.POST.get('filters', '{}')
            
            # 3. EXECUTE AI SQL AGENT WITH FILTERS
            # This function:
            # - Loads data into DuckDB
            # - Detects domain and date columns
            # - Applies filters to create filtered view
            # - Generates KPIs and charts from filtered data
            # Generate session ID for this upload
            import uuid
            session_id = str(uuid.uuid4())

            # Execute with session_id
            dashboard_data = execute_dashboard_logic(dfs, filters_json, session_id)

            # Store session_id in Django session for later use in export
            request.session['duckdb_session_id'] = session_id

            return JsonResponse(dashboard_data)
            
            return JsonResponse(dashboard_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)
@csrf_exempt
def export_powerbi(request):
    """
    Generates a ZIP file containing Universal Master CSV and Power BI template.
    """
    if request.method == 'POST':
        try:
            # Get session_id from Django session
            session_id = request.session.get('duckdb_session_id')
            
            if not session_id:
                # Fallback: Try to get from POST data
                dashboard_json = request.POST.get('dashboard_data')
                if dashboard_json:
                    dashboard_data = json.loads(dashboard_json)
                    session_id = dashboard_data.get('session_id')
            
            if not session_id:
                return JsonResponse({
                    "error": "No session found. Please upload and process your data first."
                }, status=400)
            
            # Get filters if provided
            filters_json = request.POST.get('filters', '{}')
            
            # Call the Universal Adapter function
            from .utils import get_universal_master_csv
            master_df = get_universal_master_csv(session_id, filters_json)
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # 1. Add Universal Master CSV
                csv_buffer = io.StringIO()
                master_df.to_csv(csv_buffer, index=False)
                zip_file.writestr('Universal_Master.csv', csv_buffer.getvalue())
                
                # 2. Add Power BI Template (.pbit file)
                template_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'powerbi_template.pbit')
                
                if os.path.exists(template_path):
                    with open(template_path, 'rb') as pbit_file:
                        zip_file.writestr('PowerBI_Template.pbit', pbit_file.read())
                else:
                    # If template doesn't exist, add a note
                    zip_file.writestr('TEMPLATE_MISSING.txt', 
                        'Power BI template file not found at: dashboard/static/powerbi_template.pbit\n'
                        'Please create your template and place it in this location.')
                
                # 3. Create comprehensive README
                readme = f"""POWER BI UNIVERSAL MASTER EXPORT
=====================================

WHAT'S INCLUDED:
----------------
1. Universal_Master.csv - Your complete dataset with standardized columns
2. PowerBI_Template.pbit - Pre-configured Power BI template (if available)

HOW TO USE:
-----------
STEP 1: Extract this ZIP file to a folder

STEP 2: Open Power BI Desktop

STEP 3: Load the data
   - Click "Get Data" → "Text/CSV"
   - Select "Universal_Master.csv"
   - Click "Load"

STEP 4 (Optional): Use the template
   - Open "PowerBI_Template.pbit"
   - When prompted, update the data source to point to your CSV

UNDERSTANDING THE STANDARDIZED COLUMNS:
---------------------------------------
Your data has been mapped to universal column names:

DATE COLUMN:
- STD_Date = Your date/time field
- STD_Date_Name = Tells you which original column was used

CATEGORY COLUMNS (For Slicers & Filters):
- STD_Cat1 = Primary category (e.g., Region, Product, Customer)
- STD_Cat2 = Secondary category
- STD_Cat3 = Third category
- STD_Cat4 = Fourth category
- STD_Cat5 = Fifth category
- STD_Cat1_Name through STD_Cat5_Name = Show original column names

VALUE COLUMNS (For Charts & KPIs):
- STD_Val1 = Primary numeric measure (e.g., Sales, Revenue)
- STD_Val2 = Secondary numeric measure
- STD_Val3 = Third measure
- STD_Val4 = Fourth measure
- STD_Val5 = Fifth measure
- STD_Val1_Name through STD_Val5_Name = Show original column names

CREATING DYNAMIC SLICERS:
--------------------------
1. Drag STD_Cat1 to create a slicer
2. Add a Card visual above it
3. In the Card, use this DAX formula:
   
   Slicer_Label = SELECTEDVALUE('Universal_Master'[STD_Cat1_Name], "Category 1")
   
4. This will show "Region" instead of "STD_Cat1" to your users!

CREATING DYNAMIC CHART TITLES:
-------------------------------
Use DAX measures like:

Chart_Title = 
"Total " & SELECTEDVALUE('Universal_Master'[STD_Val1_Name], "Value") & 
" by " & SELECTEDVALUE('Universal_Master'[STD_Cat1_Name], "Category")

This creates titles like: "Total Sales by Region"

NOTES:
------
- Columns marked "N/A" or "Unused" mean your dataset didn't have that many categories/values
- All slicers will work globally because everything is in ONE table
- No relationships needed - it's a single flat table!

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Rows Exported: {len(master_df):,}
"""
                zip_file.writestr('README.txt', readme)
            
            # Send ZIP file as response
            zip_buffer.seek(0)
            response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            response['Content-Disposition'] = f'attachment; filename="PowerBI_Universal_Export_{timestamp}.zip"'
            
            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)