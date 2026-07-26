class WorkspaceManager:
    def __init__(self, tenant_id="default"):
        self.tenant_id = tenant_id
    def get_workspace_prefix(self):
        return f"workspace_{self.tenant_id}_"
