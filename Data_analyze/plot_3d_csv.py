import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Read data from a csv
colname = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
z_data = pd.read_csv('81021703_1st.csv', header=None)
z_data1 = pd.read_csv('81021703_2nd.csv', header=None) + 1
z_data2 = pd.read_csv('81021703_3rd.csv', header=None) + 2

fig = go.Figure(data=[go.Surface(x=pd.array(range(1,21)),y=pd.array(range(1,26)), z=z_data.values)])
                      #go.Surface(z=z_data1.values, showscale=False, opacity=0.9),
                      #go.Surface(z=z_data2.values, showscale=False, opacity=0.9)])

fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

fig.update_layout(title='Slider sharpness', autosize=True,
                  width=1000, height=600
                  )#margin=dict(l=65, r=50, b=65, t=90)
fig.update_scenes(yaxis_autorange='reversed')

fig.show()