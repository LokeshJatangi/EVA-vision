# TD3 on Self Driving Car

### State space or Observation space
State space defines the input the TD3 Network (Actor and Critic).
The state_space defined is raw pixels of the cropped Mask image of Citymap.
The above image is obtained by : </br>
**1)** A Isosceles traingle is drawn on the Mask Image representing the car on the map.The center of the traingle is the position of the car in the map.</br>
**2)** To capture orientation of the car ,I chose the method to rotate the triangle (car) about its center, To rotate about its center, Intially a traingle is drawn at position of car with car angle = 0 (Horizontal), then rotate each point of the triangle around its center.</br>
**3)** After the Triangle is drawn ,the image is reduced to one third of its original size , Hence the center of car gets reduced and its position is also updated.</br>
**4)** To avoid failure of cropping at Walls , Padding of 20 pixels around the image is added in white background. 20 pixels padding was chosen as in the kivy env a constraint on wall is provided for 20 pixels.Even here to capture the changes of padding on the center of triangle , center of car is translated by 20 pixels.</br>
---- Image -----</br>
**5)** A 40 x 40 image is cropped and is converted to Grayscale then pixel normalization and at last converted to numpy array which needs to be fed into Actor and Critic Networks
**6)** Car states :- Position_x , Position_y , Velocity_x , Velocity_y , +Orientation , -Orientation of car .


### Action space

Action space was defined based on the gym environment Box class as it helps in capturing the Continous values and it also provides uniform sampling distribution for Bounded values.</br>
In this scenario , As in TD3 all the actions are applied simulateously in a single step on the agent while training. Hence I decided to take three independent actions as :</br>
1) Rotation of Car :</br>
    The action values provided are [-10,10] as it can turn 10 degrees to left or right.

2) Acceleration : </br>
   The action values provided are [-0.5,2] ,In this scenario a higher value of accleration increases velocity of the car in few timesteps so ,both accleration and velocity are clipped. Acceleration allows the car to move in backwards direction also. 

The critic_loss during training remained under 5000 for the above set of action_values.(In other cases critic_loss was above 1e6.

### Rewards 

Rewards considered are Living Penalty , Wall reward , Destination reward , Distance reward.</br>
Wall reward is considered when the car reaches near wall , I had given a Wall reward = **-500** and the episode would also end. </br>
Destination reward is taken when car reaches one of the five destinations and reward of **50000** is provided.</br>
Living Penalty is **-4**, which increases the negative rewards in the episode as long as it is on the map.</br>
Distance reward is **2** , if it is moving towards destination and -6 if moving away  from destination.</br>

For the above set of reward values and action values , initial losses were less than 1000.

### Training 

#### Actor : 
  Actor network has two inputs  gets the state_space of env as input in form of (1,40,40) grayscale image and states of car (position,velocity,orientation).Convolution Layers are applied with a stride of two and then a Average pooling on each layer is applied.This provides a Feature vector in 1D and car states are concatenated with feature vector to obtain action values.

#### Critic :
 Critic Network requires a Multi-Input as it has image as state , car_states and and actions from Actor Network ,So we need to combine 2D and 1D data,Here the image is converted into a 1D feature vector after few convolutions then the 1D input data (action + car_states ) is concatenated with it and forward propagation is completed for two critic networks.






