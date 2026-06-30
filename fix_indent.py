with open("app.py", encoding="utf-8") as f:
    lines = f.readlines()

out = []
in_predictions_block = False
in_bad_indent = False
for i, line in enumerate(lines):
    if (
        "if predictions:" in line
        and i + 1 < len(lines)
        and "ensemble_result =" in lines[i + 1]
    ):
        in_predictions_block = True

    if in_predictions_block and "# If SPAM, classify the threat type" in line:
        in_bad_indent = True

    if in_bad_indent:
        if (
            line.strip() == "else:"
            and i + 1 < len(lines)
            and "st.warning(" in lines[i + 1]
        ):
            out.append("                    else:\n")
            continue
        elif "st.warning(" in line and "No predictions could be generated" in line:
            out.append(
                '                        st.warning("No predictions could be generated from the ensemble models for this message.")\n'
            )
            continue
        elif (
            line.strip() == "else:"
            and i + 1 < len(lines)
            and "st.error(" in lines[i + 1]
            and "No ensemble models were loaded" in lines[i + 1]
        ):
            out.append("            else:\n")
            continue
        elif "st.error(" in line and "No ensemble models were loaded" in line:
            out.append(
                '                st.error("No ensemble models were loaded successfully. Cannot perform ensemble analysis.")\n'
            )
            in_bad_indent = False
            in_predictions_block = False
            continue
        elif line.strip() == "":
            out.append(line)
        else:
            out.append("    " + line)
    else:
        out.append(line)

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(out)
