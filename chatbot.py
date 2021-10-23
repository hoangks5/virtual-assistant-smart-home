from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import numpy
import tflearn
from stopword import stop_word 
import random
from underthesea import word_tokenize
import json
import os
from tensorflow.python.framework import ops
import re
import json
import datetime
import webbrowser as wb


# Cập nhật thêm câu hỏi và trả lời vào tệp để training
def update_json(filepath, var1, var2):
    now = datetime.datetime.now() # Gọi timenow đặt tên cho tag
    t = now.strftime("%y-%m-%d %H:%M:%S")
    with open(filepath,'r', encoding='utf-8') as fp:
        information = json.load(fp)
    information["intents"].append({
        "tag": t,
        "patterns": [var1],
        "responses": [var2],
        "context_set": ""
    })

    with open(filepath,'w',encoding='utf-8') as fp: # Thêm dữ liệu vào tệp JSON
        json.dump(information, fp,ensure_ascii=False ,indent=2,)
        



with open("training.json",encoding="utf8") as file:
    data = json.load(file)
try:
    with open("data.pickle", "ab") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
        
    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle", "ab") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load(model.tflearn)
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


    
def chat():
    print("Noi gi di may?")
    while True:
        inp = input("You: ")
        inp = stop_word(inp)
        if inp.lower() == "quit":
            break
        

        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #print(results)
        
        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    hoang = random.choice(responses)
                    if hoang == 'Google':
                        inp3 = input("Bot Dz: Bạn muốn tìm kiếm gì ? \nYou: ")
                        url=f"https://www.google.com/search?q="+inp3
                        wb.get().open(url)
                    if hoang == 'Youtube':
                        inp3 = input("Bot Dz: Bạn muốn tìm kiếm gì ? \nYou: ")
                        url=f"https://www.youtube.com/search?q="+inp3
                        wb.get().open(url)
                    if hoang == 'Facebook':
                        url=f"https://www.facebook.com/"
                        wb.get().open(url)
                    if hoang == 'Ảnh':
                        link = r"C:\Users\PC\Pictures\poster.png"
                        os.startfile(link)
                        
            print("Bot Dz: "+random.choice(responses))
        else: 
            inp2 = input("Hãy nhập câu trả lời: ")
            update_json("training.json",inp,inp2)
            
chat()