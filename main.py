from flask import Flask
from app import views
from datetime import timedelta
app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=2)

#url
app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','index',views.index)
app.add_url_rule('/faceapp','faceapp',views.faceapp)
app.add_url_rule('/faceapp/gender','gender',views.gender)
app.add_url_rule('/faceapp/gender','gender',views.gender,methods=['GET','POST'])
#run
if __name__ == '__main__':
#     app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug = True)
    
