import os
from dotenv import load_dotenv
load_dotenv()  # Loads your .env file

from azure.identity import ClientSecretCredential
# FIX 1: Added `.resources` to the import path
from azure.mgmt.resource.resources import ResourceManagementClient 
from azure.mgmt.costmanagement import CostManagementClient
from datetime import datetime

# 1. Authenticate securely
credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET")
)

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

# 2. Initialize Clients
resource_client = ResourceManagementClient(credential, subscription_id)
cost_client = CostManagementClient(credential)

# --- SKILL 1: Read Resources ---
def get_azure_resources():
    """Fetches a list of all resource groups and their resources."""
    try:
        resources = []
        # We will just list Resource Groups to keep the context small and fast
        for rg in resource_client.resource_groups.list():
            resources.append({
                "name": rg.name,
                "location": rg.location
            })
        return resources[:10] 
    except Exception as e:
        return f"Error fetching resources: {str(e)}"

# --- SKILL 2: Read Billing Info ---
# --- SKILL 2: Read Billing Info (FIXED) ---
def get_azure_billing_info():
    """Fetches the current month's Azure costs."""
    try:
        scope = f"/subscriptions/{subscription_id}"
        
        query = {
            "type": "ActualCost",
            "timeframe": "MonthToDate",
            "dataset": {
                "granularity": "Daily",
                "aggregation": {
                    # PreTaxCost is the most reliable metric name
                    "totalCost": {"name": "PreTaxCost", "function": "Sum"}
                }
            }
        }
        
        result = cost_client.query.usage(scope, query)
        
        if result.rows:
            # Look at the columns metadata to find exactly which index holds the cost
            cost_idx = next((i for i, col in enumerate(result.columns) if 'Cost' in col.name), 0)
            
            # Sum up the actual numbers from that specific column
            total_cost = sum(float(row[cost_idx]) for row in result.rows)
        else:
            total_cost = 0
            
        return f"Total cost for the current month: ${total_cost:.2f} USD"
        
    except Exception as e:
        return f"Error fetching billing info: {str(e)}"

from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.resourcegraph.models import QueryRequest, QueryRequestOptions, ResultFormat

resource_graph_client = ResourceGraphClient(credential)

def get_running_billable_resources():
    """Lists Azure resources that are currently powered on / running and incurring charges."""
    try:
        query = QueryRequest(
            subscriptions=[subscription_id],
            query="""
            Resources
            | where properties.provisioningState == 'Succeeded'
            | extend powerState = tostring(properties.extended.instanceView.powerState.code),
                     appState   = tostring(properties.state),
                     dbStatus   = tostring(properties.status)
            | extend currentState = coalesce(powerState, appState, dbStatus)
            // Keep things that are explicitly 'running' PLUS resources that are always-on (storage, networking)
            | where currentState in~ ('PowerState/running', 'Running', 'Online', 'Started', 'Ready')
                    or type in~ ('microsoft.storage/storageaccounts',
                                 'microsoft.network/virtualnetworks',
                                 'microsoft.network/publicipaddresses',
                                 'microsoft.network/loadbalancers',
                                 'microsoft.network/networksecuritygroups')
            | project name, type, location, resourceGroup, currentState
            | order by type asc, name asc
            """,
            options=QueryRequestOptions(result_format=ResultFormat.object_array)
        )
        result = resource_graph_client.resources(query)

        if not result.data:
            return "No running billable resources found."

        # Group by resource type so the LLM gets a clean summary
        grouped = {}
        for r in result.data:
            friendly_type = r['type'].split('/')[-1]   # e.g. 'virtualMachines'
            grouped.setdefault(friendly_type, []).append({
                "name": r['name'],
                "rg":   r['resourceGroup'],
                "loc":  r['location'],
                "state": r['currentState']
            })
        return grouped

    except Exception as e:
        return f"Error querying Resource Graph: {e}"

if __name__ == "__main__":
    print("--- Fetching Resources ---")
    print(get_azure_resources())
    print("\n--- Fetching Billing ---")
    print(get_azure_billing_info())
    print("\n--- Fetching Running Billable Resources ---")
    print(get_running_billable_resources())