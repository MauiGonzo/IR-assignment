import re
import xml.etree.ElementTree as ET
import pandas as pd
import os

# DIRECTORIES
FOLDER = "IR_dataset/"
QRELS = FOLDER + "qrels.robust2004.txt"
TOPICS = FOLDER + "04.testset"

# OPTIONS
LOAD_RELATIONS = True
LOAD_QUERIES = True
LOAD_DOCUMENTS = True

def get_simplified_text(text):
    combined = "".join(text.itertext()).replace('\n', ' ')
    return combined


def remove_invalid_data(xml):
    for i in range(100, 110):
        xml = xml.replace(f"<F P={i}>", f"<F P=\"{i}\">")

    xml = xml.replace("(c);", "c")
    xml = xml.replace("&|", " |")
    xml = xml.replace("&-|", "  |")
    xml = xml.replace("<3>", "")
    xml = xml.replace("</3>", "")

    xml = re.sub(r'<FIG ID=\w{1,7}-\w{1,7}-\w{1,7}-\w{1,7}>', '<FIG>', xml)
    xml = re.sub(r'<!-- (\w|\d|=|\s)*-->', '', xml)
    xml = re.sub(r'&\w{1,10}\;', '', xml)

    return xml


if LOAD_RELATIONS:

    relations = pd.read_csv(QRELS, sep=" ", header=None, names=["id_left", "X", "id_right", "label"])
    relations = relations.drop(columns="X")

    df = pd.DataFrame(relations)

    df.to_csv("relations.csv")


if LOAD_QUERIES:
    title_left = []
    desc_left = []
    ids_left = []

    with open(TOPICS, 'r', encoding="ISO-8859-1") as f:   # Reading file
        xml = f.read()

    xml = '<ROOT>\n' + xml + '</ROOT>'
    xml = xml.replace("<num> ", "_NUM_\n")
    xml = xml.replace("<num>", "_NUM_\n")
    xml = xml.replace("<title> ", "_TIT_\n")
    xml = xml.replace("<title>", "_TIT_\n")
    xml = xml.replace("<desc> ", "_DES_\n")
    xml = xml.replace("<desc>", "_DES_\n")
    xml = xml.replace("<narr> ", "_NAR_\n")
    xml = xml.replace("<narr>", "_NAR_\n")
    xml = xml.replace("&", " and ")

    root = ET.fromstring(xml)

    for doc in root:
        lines = doc.text.splitlines()
        number = False
        title = False
        desc = False
        number_str = ""
        title_str = ""
        desc_str = ""
        for line in lines:
            # print(line)

            if re.search("_NUM_", line):
                number = True
                title = False
                desc = False
                continue
            if re.search("_TIT_", line):
                number = False
                title = True
                desc = False
                continue
            if re.search("_DES_", line):
                number = False
                title = False
                desc = True
                continue
            if re.search("_NAR_", line):
                number = False
                title = False
                desc = False
                continue

            if number:
                if re.findall("\d+", line):
                    id = re.findall("\d+", line)[0]
                    number = False
            if title:
                title_str = title_str + " " + line
            if desc:
                if not re.search("Description:", line):
                    desc_str = desc_str + " " + line

        print(id)
        print(title_str)
        print(desc_str)
        print("")
        title_left.append(str(title_str.replace("\"", "")))
        desc_left.append(str(desc_str))
        ids_left.append(id)


    data = {
        "text_left" : title_left
    }

    df = pd.DataFrame(data, index = ids_left)

    df.to_csv("title_querries.csv")

    data = {
        "text_left" : desc_left
    }

    df = pd.DataFrame(data, index = ids_left)

    df.to_csv("desc_querries.csv")


if LOAD_DOCUMENTS:
    text_right = []
    ids = []

    # FR94 DATABASE

    FT_FOLDER = FOLDER + "ft/"

    for file in os.listdir(FT_FOLDER):
        print("Outer")
        print(file)

        if file.startswith("read") or file.startswith("."):
            continue

        combined = os.path.join(FT_FOLDER, file)
        for inner_file in os.listdir(combined):
            print(inner_file)

            if inner_file.startswith("."):
                continue

            with open(os.path.join(combined, inner_file), 'r', encoding="ISO-8859-1") as f:   # Reading file
                xml = f.read()

            xml = '<ROOT>' + xml + '</ROOT>'

            xml = remove_invalid_data(xml)

            root = ET.fromstring(xml)

            for doc in root:
                id = doc.find('DOCNO').text.strip()
                text = get_simplified_text(doc)
                ids.append(id)
                text_right.append(text)


    # FR94 DATABASE

    FR94_FOLDER = FOLDER + "fr94/"

    for file in os.listdir(FR94_FOLDER):
        print("Outer")
        print(file)

        if file.endswith("gz") or file.endswith("aux") or file.startswith("."):
            continue

        combined = os.path.join(FR94_FOLDER, file)
        for inner_file in os.listdir(combined):
            print(inner_file)

            with open(os.path.join(combined, inner_file), 'r', encoding="ISO-8859-1") as f:   # Reading file
                xml = f.read()

            xml = '<ROOT>' + xml + '</ROOT>'

            xml = remove_invalid_data(xml)

            root = ET.fromstring(xml)

            for doc in root:
                id = doc.find('DOCNO').text.strip()
                text = get_simplified_text(doc)
                ids.append(id)
                text_right.append(text)

    # FBIS DATABASE

    FBIS_FOLDER = FOLDER + "fbis/"

    for file in os.listdir(FBIS_FOLDER):
        print(file)

        if file.startswith("read") or file.startswith("."):
            continue

        if file.startswith('fb'):

            with open(os.path.join(FBIS_FOLDER, file), 'r', encoding="ISO-8859-1") as f:   # Reading file
                xml = f.read()

            xml = '<ROOT>' + xml + '</ROOT>'

            xml = remove_invalid_data(xml)

            root = ET.fromstring(xml)

            for doc in root:
                id = doc.find('DOCNO').text.strip()
                text = get_simplified_text(doc)
                ids.append(id)
                text_right.append(text)

    # LA TIMES DATABASE

    LA_FOLDER = FOLDER + "latimes/"

    for file in os.listdir(LA_FOLDER):
        print(file)

        if file.startswith("read") or file.startswith("."):
            continue

        if file.startswith('la'):

            with open(os.path.join(LA_FOLDER, file), 'r') as f:
                xml = f.read()

            xml = '<ROOT>' + xml + '</ROOT>'
            root = ET.fromstring(xml)

            for doc in root:
                id = doc.find('DOCNO').text.strip()
                text = get_simplified_text(doc)
                ids.append(id)
                text_right.append(text)

    data = {
        "text_right" : text_right
    }

    df = pd.DataFrame(data, index = ids)

    df.to_csv("full.csv")
