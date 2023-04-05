import gradio as gr
import paperqa
import pickle
import pandas as pd
from pathlib import Path
import requests
import zipfile
import io
import tempfile
import os


css_style = """

.gradio-container {
    font-family: "IBM Plex Mono";
}
"""


def request_pathname(files, data, openai_api_key):
    if files is None:
        return [[]]
    for file in files:
        # make sure we're not duplicating things in the dataset
        if file.name in [x[0] for x in data]:
            continue
        data.append([file.name, None, None])
    return [[len(data), 0]], data, data, validate_dataset(pd.DataFrame(data), openai_api_key)


def validate_dataset(dataset, openapi):
    docs_ready = dataset.iloc[-1, 0] != ""
    if docs_ready and type(openapi) is str and len(openapi) > 0:
        return "✨Ready✨"
    elif docs_ready:
        return "⚠️Waiting for key⚠️"
    elif type(openapi) is str and len(openapi) > 0:
        return "⚠️Waiting for documents⚠️"
    else:
        return "⚠️Waiting for documents and key⚠️"


def make_stats(docs):
    return [[len(docs.doc_previews), sum([x[0] for x in docs.doc_previews])]]


# , progress=gr.Progress()):
def do_ask(question, button, openapi, dataset, length, do_marg, k, max_sources, docs):
    passages = ""
    docs_ready = dataset.iloc[-1, 0] != ""
    if button == "✨Ready✨" and type(openapi) is str and len(openapi) > 0 and docs_ready:
        os.environ['OPENAI_API_KEY'] = openapi.strip()
        if docs is None:
            docs = paperqa.Docs()
        # dataset is pandas dataframe
        for _, row in dataset.iterrows():
            try:
                docs.add(row['filepath'], row['citation string'],
                         key=row['key'], disable_check=True)
                yield "", "", "", docs, make_stats(docs)
            except Exception as e:
                pass
    else:
        yield "", "", "", docs, [[0, 0]]
    #progress(0, "Building Index...")
    docs._build_faiss_index()
    #progress(0.25, "Querying...")
    for i, result in enumerate(docs.query_gen(question,
                                              length_prompt=f'use {length:d} words',
                                              marginal_relevance=do_marg,
                                              k=k, max_sources=max_sources)):
        #progress(0.25 + 0.1 * i, "Generating Context" + str(i))
        yield result.formatted_answer, result.context, passages, docs,  make_stats(docs)
    #progress(1.0, "Done!")
    # format the passages
    for i, (key, passage) in enumerate(result.passages.items()):
        passages += f'Disabled for now'
    yield result.formatted_answer, result.context, passages, docs,  make_stats(docs)


def download_repo(gh_repo, data, openai_api_key, pbar=gr.Progress()):
    # download zipped version of repo
    r = requests.get(f'https://api.github.com/repos/{gh_repo}/zipball')
    if r.status_code == 200:
        pbar(1, 'Downloaded')

        # iterate through files in zip
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for i, f in enumerate(z.namelist()):
                # skip directories
                if f.endswith('/'):
                    continue
                # try to read as plaintext (skip binary files)
                try:
                    text = z.read(f).decode('utf-8')
                except UnicodeDecodeError:
                    continue
                # check if it's bigger than 100kb or smaller than 10 bytes
                if len(text) > 1e5 or len(text) < 10:
                    continue
                # have to save to temporary file so we have a path
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(text.encode('utf-8'))
                    tmp.flush()
                    path = tmp.name
                    # strip off the first directory of f
                    rel_path = '/'.join(f.split('/')[1:])
                    key = os.path.basename(f)
                    citation = f'[{rel_path}](https://github.com/{gh_repo}/tree/main/{rel_path})'
                    if path in [x[0] for x in data]:
                        continue
                    data.append([path, citation, key])
                    yield [[len(data), 0]], data, data, validate_dataset(pd.DataFrame(data), openai_api_key)
                pbar(int((i+1)/len(z.namelist()) * 99),
                     f'Added {f}')
        pbar(100, 'Done')
    else:
        raise ValueError('Unknown Github Repo')
    return data


with gr.Blocks(css=css_style) as demo:

    docs = gr.State(None)
    data = gr.State([])
    openai_api_key = gr.State('')

    gr.Markdown(f"""
    # Document Question and Answer (v{paperqa.__version__})

    *By Andrew White ([@andrewwhite01](https://twitter.com/andrewwhite01))*

    This tool will enable asking questions of your uploaded text, PDF documents,
    or scrape github repos.
    It uses OpenAI's GPT models and thus you must enter your API key below. This
    tool is under active development and currently uses many tokens - up to 10,000
    for a single query. That is $0.10-0.20 per query, so please be careful!

    * [PaperQA](https://github.com/whitead/paper-qa) is the code used to build this tool.
    * [langchain](https://github.com/hwchase17/langchain) is the main library this tool utilizes.

    1. Enter API Key ([What is that?](https://platform.openai.com/account/api-keys))
    2. Upload your documents
    3. Ask a questions
    """)
    openai_api_key = gr.Textbox(
        label="OpenAI API Key", placeholder="sk-...", type="password")
    with gr.Tab('File Upload'):
        uploaded_files = gr.File(
            label="Your Documents Upload (PDF or txt)", file_count="multiple", )
    with gr.Tab('Github Repo'):
        gh_repo = gr.Textbox(
            label="Github Repo", placeholder="whitead/paper-qa")
        download = gr.Button("Download Repo")

    with gr.Accordion("See Docs:", open=False):
        dataset = gr.Dataframe(
            headers=["filepath", "citation string", "key"],
            datatype=["str", "str", "str"],
            col_count=(3, "fixed"),
            interactive=False,
            label="Documents and Citations",
            overflow_row_behaviour='paginate',
            max_rows=5
        )
    buildb = gr.Textbox("⚠️Waiting for documents and key...",
                        label="Status", interactive=False, show_label=True,
                        max_lines=1)
    stats = gr.Dataframe(headers=['Docs', 'Chunks'],
                         datatype=['number', 'number'],
                         col_count=(2, "fixed"),
                         interactive=False,
                         label="Doc Stats")
    openai_api_key.change(validate_dataset, inputs=[
                          dataset, openai_api_key], outputs=[buildb])
    dataset.change(validate_dataset, inputs=[
                   dataset, openai_api_key], outputs=[buildb])
    uploaded_files.change(request_pathname, inputs=[
                          uploaded_files, data, openai_api_key], outputs=[stats, data, dataset, buildb])
    download.click(fn=download_repo, inputs=[
                   gh_repo, data, openai_api_key], outputs=[stats, data, dataset, buildb])
    query = gr.Textbox(
        placeholder="Enter your question here...", label="Question")
    with gr.Row():
        length = gr.Slider(25, 200, value=100, step=5,
                           label='Words in answer')
        marg = gr.Checkbox(True, label='Max marginal relevance')
        k = gr.Slider(1, 20, value=10, step=1,
                      label='Chunks to examine')
        sources = gr.Slider(1, 10, value=5, step=1,
                            label='Contexts to include')

    ask = gr.Button("Ask Question")
    answer = gr.Markdown(label="Answer")
    with gr.Accordion("Context", open=True):
        context = gr.Markdown(label="Context")

    with gr.Accordion("Raw Text", open=False):
        passages = gr.Markdown(label="Passages")
    ask.click(fn=do_ask, inputs=[query, buildb,
                                 openai_api_key, dataset,
                                 length, marg, k, sources,
                                 docs], outputs=[answer, context, passages, docs, stats])

demo.queue(concurrency_count=20)
demo.launch(show_error=True)
