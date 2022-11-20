import os
from flask import Flask, render_template, redirect, url_for,request, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField,validators,SelectField
from wtforms.widgets import TextArea
from werkzeug.datastructures import MultiDict
import datetime
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from model import knn_comparison

app= Flask(__name__)
app.config['SECRET_KEY'] = 'mykey'



class SubmitForm(FlaskForm):
    K = IntegerField(u'K : ')
    data_points = SelectField(u'Points distribution:',
                choices=[('d1', 'Ushape'),('d2','ConcerticCircle1'),('d3','ConcerticCircle2'),('d4','LinearSep'),('d5','Outlier'),('d6','Overlap'),('d7','XOR'),('d8','Spirals'),('d9','Random')])
    submit = SubmitField('Classify')


@app.route('/', methods=['GET','POST'])
def index():
    form = SubmitForm()
    if form.validate_on_submit():
        session["K"] = form.K.data
        session["points_dist"] = form.data_points.data
    return render_template("home.html", form = form)

@app.route('/plot.png')
def create_figure():
    dict = {'d1':'1.ushape.csv','d2':'2.concerticcir1.csv','d3':'3.concertriccir2.csv','d4':'4.linearsep.csv','d5':'5.outlier.csv','d6':'6.overlap.csv','d7':'7.xor.csv','d8':'8.twospirals.csv','d9':'9.random.csv'}
    d = session["points_dist"]
    path = 'data/' + dict.get(d)
    K = session["K"]
    data = np.genfromtxt(path, delimiter=',')
    fig = knn_comparison(data, K)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    plt.close()
    return Response(output.getvalue(), mimetype="image/png")




if __name__ == '__main__':
	app.run(debug=True)
