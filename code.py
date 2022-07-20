import pandas as pd
import torch
import torch.nn as nn #to crreate neural network
import tez  #simple and fast library to make trianing easy
from sklearn import model_selection
from sklearn import metrics , preprocessing  # preprocessing to encode users and movies 
import numpy as np



class MovieDatasets():
  def __init__(self, users , movies , ratings):
    self.users = users
    self.movies = movies
    self.ratings = ratings

  def __len__(self):
    return len(self.users)

  def __getitem__(self,item):
     users = self.users[items]
     movie = self.movies[items]
     ratings = self.ratings[items]

     return {"users":torch.tensor(user,dtype= torch.long),
             "movies":torch.tensor(movies,dtype= torch.long)
             "ratings":torch.tensor(ratings,dtype= torch.float)
             }
  #Recommendation System   
class RecSysModel(tez.model):
  def __init__(self,num_users,num_movies):
    super().__init__()
    self.user_embed = nn.Embedding(num_users , 32)
    self.movie_embed = nn.Embedding(num_movies , 32)
    self.out = nn.Linear(64 ,1)  #32+32 and out 1 value
    self.step_schedular_after = "epoch"

   def fetch_optimizer(self):
     opt = torch.optim.Adam(self.parameters(),lr=1e-3)  
     return opt

   def fetch_schedular(self):
     sch = torch.optim.lr_schedular.StepLR(self.optimizer,step_size =3 , gamma = 0.7)  #gamma defines how far the influence of a single training example reaches\
     return sch


  def monitor _metrics(self, output ,rating):
    output = output.detach().cpu,numpy()  #will detach it from cpu and using numpy to convert it
    rating = rating.detach().cpu,numpy()
    return {
        'rmse':np.sqrt(metrics.mean_squared_error(rating , output))
    }




  def forward(self, users , movie , rating = None):
    users_embeds = self.users_embeds(users)
    users_embeds = self.users_embeds(users)
    output = torch.cat([user_embeds , movie_embeds], dim = 1)  #concatenate
    output = self.out(output)
    loss = nn.MSELoss()(output , rating.view(-1 ,1)) #n *1 row 
    calc_merics = self.monitor_metrics(output , ratings.view(-1,1))
    return output , loss , calc_metrics
    
def train():
  df = pd.read_csv("")
  #df.user.nunique()
  #df.movie.nunique()
  #df.shape
  #df.rating.value_counts()  #check the last column
  lbl_user = preprocesing.LabelEncoder()
  lbl_movie = preprocesing.LabelEncoder()

  df.user = lbl_user.fit_transform(df.user.values)
  df.movie = lbl_movie.fit_transform(df.user.values)



  df_train , df_valid = model_selection.train_test_split(df,test_size=0.2,random_state=42,statify =df.rating.value )
  train_dataset = MovieDatasets(users=df_train.users.values ,movies= df_train.movies.values ,ratings= df_train.ratings.values )
  valid_dataset = MovieDatasets(users=df_valid.users.values ,movies= df_valid.movies.values ,ratings= df_valid.ratings.values )
model = RecSys(num_users=len(lbl_user.classes_),len(lbl_movie.classes_))
model.fit(
    train_dataset, valid_dataset , train_bs = 1024 , valid_bs = 1024 , fp16 = True , 
)
#fp16 = true requires low memory for training and shortens the training time 


if __name__ == "main":
  train() #run train function





