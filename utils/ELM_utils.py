import os
import re
import h5py
import torch
import argparse
import torch.nn.functional as F

import numpy as np


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'
        
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if " " in text: text = text.replace(". ",".<stop>")
    if "?" in text: text = text.replace("? ","?<stop>")
    if "!" in text: text = text.replace("! ","!<stop>")
    if "<prd>" in text: text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences
    
def preprocess_report(report, text_sample_mode, requested_headings, simple=False, sampling=True, include_heading=False, prefix="Clinical EEG report: ", prefiltered=False):
    """
    Filter report depending on requested headings.
    INPUT
    report: str
    text_sample_mode: str (report, paragraph, sentence)
    requested_headings: list of str (CLINICAL_HISTORY, ...,). ['all'] includes every heading
    mode: Boolean (True: sampling, False: readout). Whether to return 1 paragraph/sentence, or all of them.
    OUTPUT
    list of strings, and boolean indicating whether its returning the requested headings
    """
    
    delimiters = [
    "DATE OF STUDY",
    "START DATE OF STUDY",
    "END DATE OF STUDY",
    "REASON FOR STUDY",
    "DAY",
    "RECORDING ENVIRONMENT",
    "EVENT",
    "EEG TYPE",
    "HISTORY", 
    "CLINICAL HISTORY", 
    "MEDICATIONS", 
    "AED",
    "INTRODUCTION", 
    "SEDATION", 
    "SEIZURE TIME",
    "SEIZURES",
    "ABNORMAL DISCHARGES",
    "TECHNIQUE",
    "TECHNICAL ISSUE",
    "TECHNICAL PROBLEM",
    "TECHNICAL DIFFICULTIES",
    "DESCRIPTION OF THE RECORD", 
    "DESCRIPTION OF RECORD",
    "DESCRIPTION",
    "EEG BACKGROUND", 
    "BACKGROUND",
    "EPILEPTIFORM ACTIVITY", 
    "OTHER PAROXYSMAL ACTIVITY (NON-EPILEPTIFORM)",
    "ACTIVATION PROCEDURES", 
    "INDUCTION PROCEDURES",
    "EVENTS", 
    "HEART RATE", 
    "HR",
    "CONDITIONS OF THE RECORDING",
    "RANDOM WAKEFULNESS AND SLEEP",
    "IMPRESSION", 
    "CLINICAL CORRELATION", 
    "CORRELATION",
    "CONCLUSION",
    "SUMMARY",
    "SUMMARY OF FINDINGS",
    "DIAGNOSIS", 
    "INTERPRETATION"
    ]
    
    requests_found = False
    
    report = report.replace("\u2028", "\n")

    if prefiltered:
        pieces = report.split("\n\n\n\n")[1:]
        if sampling:
            sentence = np.random.choice(pieces)
            return [prefix + sentence.replace("  ", " ").strip()], True
        else:
            return pieces, requests_found

    # Simple mode just dumps the report back with minimal preprocessing
    if simple:
        report = report.replace("Clinical EEG report: ", "")
        return [prefix + " " + report.replace("\n", " ").strip()], requests_found
    
    # Catch rare, very short unstructured reports
    # if len(report) < 200:
    #     return [prefix + " " + report.replace("\n", " ").strip()], requests_found
    
    # Replace double space with single space as this can mess up header detection.
    report = report.replace("  ", " ")
    
    # Remove any random text from report that may precede a header.
    preamble = report.split("\n")[0]
    valid_header = any(s in preamble.upper() for s in delimiters)
    if valid_header == False:
        report = report.replace(preamble, "")
    
    # First, split into paragraphs.
    pattern = '|'.join(map(re.escape, delimiters))

    sections = re.split(f'(^|\n)({pattern})', report, flags=re.IGNORECASE|re.MULTILINE)

    # Remove empty strings from the sections list
    sections = [section.strip() for section in sections if section.strip()]
    
    # If the first section is actually random text and not a header, it'll complicate the rest of the code. 
    # Thus, delete sections until we find a header.
    valid_start = False
    i = 0
    while (valid_start==False) and (len(sections)>0):
        valid_start = sections[i].upper().startswith(tuple(delimiters))
        if valid_start == False:
            sections.pop(0)

    # Now that we have a valid header, we assume the repeating structure:
    # header -> content/paragraph -> header ...
    headings, paragraphs = [], []
    for i in range(0, len(sections), 2):
        heading = sections[i].strip()
        content = sections[i+1] if i+1 < len(sections) else ""
        content = content.replace(':', '').strip()
        if include_heading:
            paragraph = f"{heading}:" + "\n" + content
        else:
            paragraph = content
        headings.append(heading)
        paragraphs.append(paragraph)
        
    # If no paragraphs are detected, we'll move from paragraph to sentence
    if len(paragraphs) == 0: 
        all_sentences = split_into_sentences(report.replace("\n", " "))
        if sampling:
            sentence = np.random.choice(all_sentences)
            return [prefix + sentence.replace("  ", " ").strip()], requests_found
        else:
            return [prefix + sentence.replace("  ", " ").strip() for sentence in all_sentences], requests_found
            
    # Reduce to include paragraphs only with requested headings
    if requested_headings == ["all"]:
        mask = np.array([True]*len(paragraphs))
    else:
        #mask = np.isin(headings, requested_headings)
        mask = []
        for head in headings:
            mask.append(any([req_head in head.upper() for req_head in requested_headings]))
        mask = np.array(mask)
        
    requests_found = True if mask.sum() else False
    if mask.sum() == 0: # If no requested headings are found, select a random one.
        # assert sampling==True, "No requested headings found while in readout mode."
        if sampling:
            sample = np.random.randint(0, len(paragraphs))
            mask[sample] = True
        else:
            print("No requested headings found while in readout mode. Return empty list.")
            return [], requests_found
    
    selected_paragraphs = np.array(paragraphs)[mask]
    selected_headings = np.array(headings)[mask]
           
    if text_sample_mode == "report":
        reduced_report = f"""{prefix}"""
        for s in selected_paragraphs:
            reduced_report += " " + s   
        return [reduced_report.replace("\n", " ").strip()], requests_found
    
    if not sampling: # return all relevant paragraphs or sentences
        if text_sample_mode == "paragraph":
            return [prefix + paragraph.replace("\n", " ").replace("  ", " ").strip() for paragraph in selected_paragraphs], requests_found
        elif text_sample_mode == "sentence":
            all_sentences = []
            for paragraph in selected_paragraphs:
                paragraph_sentences = split_into_sentences(paragraph.replace("\n", " "))
                all_sentences.extend([prefix + sentence.replace("  ", " ").strip() for sentence in paragraph_sentences])
            return all_sentences, requests_found
        
    assert sampling==True   
    # for both paragraph or sentence modes, sample one paragraph
    index = np.random.randint(0, len(selected_paragraphs))
    paragraph = selected_paragraphs[index]
    heading = selected_headings[index]    
        
    if text_sample_mode == "paragraph":
        paragraph = prefix + " " + paragraph
        return [paragraph.replace("\n", " ").replace("  ", " ").strip()], requests_found
    
    elif text_sample_mode == "sentence":
        sentence = np.random.choice(split_into_sentences(paragraph.replace("\n", " ")))
        sentence = sentence.replace(heading, "")
        sentence = prefix + " " + sentence
        return [sentence.replace("  ", " ").strip()], requests_found
    
def cosine_similarity(emb_x, emb_y):
    emb_x = F.normalize(emb_x, dim=1)
    emb_y = F.normalize(emb_y, dim=1)
    return torch.mm(emb_x, emb_y.t())


def LLM_summarization(token):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import chardet
    
    custom_cache_dir = '/path/to/models_cache/'
    url = "meta-llama/Meta-Llama-3-8B-Instruct"

    max_length = 1000
    trunc = True
    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(url, cache_dir = custom_cache_dir, torch_dtype=torch.bfloat16,
                                                token="token").to(device) ##
    tokenizer = AutoTokenizer.from_pretrained(url, cache_dir = custom_cache_dir, token="token",
                                            trunctation_side="left", max_length=max_length, truncation=trunc)
    tokenizer.truncation_side = "left" # Required!
    
    
    path = "/path/to/text/reports/"
    save_path = "/path/to/save/"
    reports = os.listdir(path)

    number_of_tokens = []

    for i, report in enumerate(reports):
        print("Report #", i, " loading:", path+report)
        
        try:
            with open(path + report,'r',newline='') as rf:
                text = rf.read()
        except:
            with open(path + report, 'rb') as file:
                raw_data = file.read(10000)  # Read the first 10000 bytes to guess the encoding
                result = chardet.detect(raw_data)
            try:
                with open(path + report, 'r', encoding=result["encoding"]) as rf:
                    text = rf.read()
            except:
                print("*"*40)
                print("SKIPPING!")
                print("*"*40)
                continue
        
        text = text.replace("\n", " ")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant adhering strictly to instructions."},
            {"role": "user", "content": f"""You are provided a clinical EEG report. Conclude whether the EEG that it described is normal, abnormal, or whether this is unclear. 
             
             Include absolutely no text beyond this conclusion. Do not provide your reasoning. Your answer should be structured as such:
             
            "Conclusion: Abnormal" or "Conclusion: Normal" or, if the EEG status is unclear, write "Conclusion: Unclear".

        Here is the report:
                
        {text}"""},
        ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                max_length=max_length, truncation=True,).to(device)

        terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            eos_token_id=terminators,
            do_sample=False,
        )
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)

        # Count output tokens
        output_tokens = tokenizer.encode(decoded_response)
        output_token_count = len(output_tokens)
        
        number_of_tokens.append(output_token_count)
        
        with open(save_path + report,'w',newline='') as wf:
            wf.write("Clinical EEG report: " + decoded_response)
        
    np.save(save_path + "tokens.npy", number_of_tokens)
    

def LLM_conclusion(start, end):
    import chardet
    from llama_cpp import Llama 

    llm = Llama(
            model_path="/path/to/model",
            chat_format="llama-3",
            n_gpu_layers=15000,
            n_ctx=8192)

    path = "/path/to/text/reports/"
    save_path = "/path/to/save/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    reports = os.listdir(path)

    def cut_report(llm, report: str, max_tok: int) -> str:
        n_tok = len(llm.tokenize(report.encode('utf-8')))
        n_char = len(report.encode('utf-8'))
        
        if n_tok > max_tok:
            frac = max_tok / n_tok
            start = (1 - frac)  * n_char
            report = report[int(start):]
            print("reduced length", len(report.encode('utf-8')))
        else:
            return report

    for i, report in enumerate(reports[start:end]):
        print("Report #", i, " loading:", path+report)
        
        try:
            with open(path + report,'r',newline='') as rf:
                text = rf.read()
        except:
            with open(path + report, 'rb') as file: 
                raw_data = file.read(10000)  # Read the first 10000 bytes to guess the encoding
                result = chardet.detect(raw_data)
            try:
                with open(path + report, 'r', encoding=result["encoding"]) as rf:
                    text = rf.read()
            except:
                print("*"*40)
                print("SKIPPING!")
                print("*"*40)
                continue
        
        text = preprocess_report(text, simple=True, text_sample_mode="report", requested_headings=["all"], include_heading=True, prefix='', sampling=False)
        
        text = text[0][0].strip() 
        text = cut_report(llm, text, max_tok=7000)

        out = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are a helpful assistant adhering strictly to instructions."},
            {"role": "user", "content": f"""
You are provided a clinical EEG report. Your task is to derive whether the described EEG is normal (without signs of pathology) or abnormal (signs of pathology). 

To do so, strictly stick to the following format. Do not output anything besides the two requested answering fields.

First, succinctly reason about the observations and interpretation of the EEG status by the physician as documented in the report. For this, use the "Reasoning" field below.
Finally, you write your conclusion in the "Conclusion" field below. Your ONLY options are Normal, Abnormal, or Uncertain (in case the EEG status is unclear from the report).

* Reasoning *: [Your reasoning goes here]

* Conclusion *: [Normal, Abnormal, Uncertain]

Here is the report.

[Start]

{text}

[End]

"""},
        ],
        seed=i
        )
        output = out["choices"][0]["message"]["content"]

        file_path = os.path.join(save_path, report)
        with open(file_path,'w',newline='') as wf:
            wf.write(output)



            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-arg1", "--argument1", type=int, dest="arg1")
    parser.add_argument("-arg2", "--argument2", type=int, dest="arg2")
    args = parser.parse_args()

    #generate_syn_data(args.arg1, args.arg2)
