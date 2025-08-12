import ssl

import httpx

url = "https://clinicaltrials.gov/api/v2/studies?query.term=Covid-19+vaccines&fields=NCTId%2COfficialTitle&pageSize=1000&countTotal=true&sort=%40relevance"
headers = {
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Host": "clinicaltrials.gov",
    "User-Agent": "python-httpx/0.28.1",
}

context = ssl.create_default_context()
context.maximum_version = ssl.TLSVersion.TLSv1_2

with httpx.Client(verify=context) as client:
    response = client.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Text: {response.text}")
