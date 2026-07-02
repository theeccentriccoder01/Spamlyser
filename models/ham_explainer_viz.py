def format_ham_explanation(tokens: list[str], weights: list[float]) -> str:
    """Formats hamminess tokens and weights into markdown explanation."""
    lines = ["### Ham Token Analysis"]
    for t, w in zip(tokens, weights):
        lines.append(f"- **{t}**: {w:.4f} (influence towards ham)")
    return "\n".join(lines)
