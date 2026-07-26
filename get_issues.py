import json
import urllib.request

req = urllib.request.Request(
    "https://api.github.com/repos/theeccentriccoder01/Spamlyser/issues?state=all&per_page=100"
)
req.add_header("User-Agent", "Mozilla/5.0")
try:
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode("utf-8"))
    for issue in data:
        print(f"[{issue['state']}] #{issue['number']}: {issue['title']}")
        # if it has labels, print them
        labels = [l["name"] for l in issue.get("labels", [])]
        if labels:
            print(f"  Labels: {', '.join(labels)}")
except Exception as e:
    print(f"Error: {e}")
