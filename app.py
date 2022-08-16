from flask import Flask,request,render_template
import pandas as pd, numpy as np
import pickle, warnings

def get_data(fn,f1,f2):
    da = dict(f1)
    da['age'] = int(da['age'])
    if(da['gender'] == 'male'):
        da['gender'] = 'm'
    elif(da['gender'] == 'female'):
        da['gender'] = 'f'
    di = dict(f2)   

    l1,l2 = [],[]
    
    if(fn=='adult_Asd'):        
        l1 = [1,7,8,0]
        l2 = [2, 3, 4, 5, 6, 9]
        del da['Family_mem_with_ASD']
        m = "Pickle\\Model1.pkl"
        e = "Pickle\\encoder1.pkl"
    elif(fn== 'asd_child'):
        l1 = [1,5,7,0]
        l2 = [2, 3, 4, 6, 8, 9]        
        del da['Family_mem_with_ASD']
        m = "Pickle\\Model2.pkl"
        e = "Pickle\\encoder2.pkl"
    elif(fn=='asd_toddlers'):
        l1 = [1,2,3,4,5,6,7,8]
        l2 = [0]        
        del da['country']
        m = "Pickle\\Model3.pkl"
        e = "Pickle\\encoder3.pkl"

    score = []
    for i,j in di.items():
        if("Agree" in j):
            if(int(i[-1]) in l1):
                score.append(1)
            elif(int(i[-1]) in l2):
                score.append(0)
        elif("Disagree" in j):
            if(int(i[-1]) in l2):
                score.append(1)
            elif(int(i[-1]) in l1):
                score.append(0)

    return da,score,m,e

def process_data(da,score,e):    
    co = ['gender','ethnicity','jaundice','country','Who completed the test','Class/ASD','Family_mem_with_ASD']
    enc = pickle.load(open(e, "rb"))
    data = pd.DataFrame([da])
    data = data.apply(lambda x: enc[x.name].transform(x) if x.name in co else x)
    final_data = score + list(data.values[0])
    return final_data

def predict(data,m):
    model = pickle.load(open(m, "rb"))
    data = np.array(data).reshape(1,-1)
    pred = model.predict(data)[0]
    if(pred):
        res="ASD"
    else:
        res="NO ASD"    
    return res, pred

obj = {'l':"",'f':""}
warnings.filterwarnings("ignore")

app= Flask(__name__)
@app.route('/')
def home():
    return render_template("form.html")

@app.route('/test',methods=['POST'])
def test():
    f = request.form
    age = int(f['age'])
    if (age < 3): 
        l= 'asd_toddlers'
    elif(age > 3 and age < 12):           
        l= 'asd_child'
    else: 
        l= 'adult_Asd'
    
    obj['l'] = l
    obj['f'] = f
    return render_template(l+".html")

@app.route('/result',methods=['POST'])
def result():        
    da,score,m,e = get_data(obj['l'],obj['f'],request.form)
    final_data = process_data(da,score,e)
    res,pred= predict(final_data,m)
    return render_template("new.html",r=res,pred=pred)

@app.route('/org')
def org():
    return render_template("list_org.html")

if __name__=='__main__':
    app.run(debug=True)
