from models.workspace_manager import WorkspaceManager

def test_workspace_manager():
    mgr = WorkspaceManager("user123")
    assert mgr.get_workspace_prefix() == "workspace_user123_"
