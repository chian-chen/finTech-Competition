import os
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具



def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {
        int(file.replace(".pdf", "")): read_pdf(os.path.join(source_path, file))
        for file in tqdm(masked_file_ls)
    }  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典

    return corpus_dict


def process_text(text):
    return text


def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0] : page_infos[1]] if page_infos else pdf.pages
    pdf_text = ""

    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += process_text(text)

    pdf.close()  # 關閉PDF文件

    return pdf_text