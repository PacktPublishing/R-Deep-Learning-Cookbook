##############################   REINFORCEMENT LEARNING : Q LEARNING  ##############

## Define States and Actions
actions <- c("up", "left", "down", "right")
states <- c("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16")

## Define transition function
transitionStateAction <- function(state, action) {
  # The default state is the existing state in case of constrained action
  next_state <- state
  if (state == "s1" && action == "down") next_state <- "s2"
  if (state == "s1" && action == "right") next_state <- "s5"
  if (state == "s2" && action == "up") next_state <- "s1"
  if (state == "s2" && action == "right") next_state <- "s6"
  if (state == "s3" && action == "right") next_state <- "s7"
  if (state == "s3" && action == "down") next_state <- "s4"
  if (state == "s4" && action == "up") next_state <- "s3"
  if (state == "s5" && action == "right") next_state <- "s9"
  if (state == "s5" && action == "down") next_state <- "s6"
  if (state == "s5" && action == "left") next_state <- "s1"
  if (state == "s6" && action == "up") next_state <- "s5"
  if (state == "s6" && action == "down") next_state <- "s7"
  if (state == "s6" && action == "left") next_state <- "s2"
  if (state == "s7" && action == "up") next_state <- "s6"
  if (state == "s7" && action == "right") next_state <- "s11"
  if (state == "s7" && action == "down") next_state <- "s8"
  if (state == "s7" && action == "left") next_state <- "s3"
  if (state == "s8" && action == "up") next_state <- "s7"
  if (state == "s8" && action == "right") next_state <- "s12"
  if (state == "s9" && action == "right") next_state <- "s13"
  if (state == "s9" && action == "down") next_state <- "s10"
  if (state == "s9" && action == "left") next_state <- "s5"
  if (state == "s10" && action == "up") next_state <- "s9"
  if (state == "s10" && action == "right") next_state <- "s14"
  if (state == "s10" && action == "down") next_state <- "s11"
  if (state == "s11" && action == "up") next_state <- "s10"
  if (state == "s11" && action == "right") next_state <- "s15"
  if (state == "s11" && action == "left") next_state <- "s7"
  if (state == "s12" && action == "right") next_state <- "s16"
  if (state == "s12" && action == "left") next_state <- "s8"
  if (state == "s13" && action == "down") next_state <- "s14"
  if (state == "s13" && action == "left") next_state <- "s9"
  if (state == "s14" && action == "up") next_state <- "s13"
  if (state == "s14" && action == "down") next_state <- "s15"
  if (state == "s14" && action == "left") next_state <- "s10"
  if (state == "s15" && action == "up") next_state <- "s14"
  if (state == "s15" && action == "down") next_state <- "s16"
  if (state == "s15" && action == "left") next_state <- "s11"
  if (state == "s16" && action == "up") next_state <- "s15"
  if (state == "s16" && action == "left") next_state <- "s12"
  # Calculate reward
  if (next_state == "s15") {
    reward <- 100
  } else {
    reward <- -1
  }
  
  return(list("state"=next_state, "reward"=reward))
}

## Define Q-Learning function
Qlearning <- function(n, initState, termState,
                      epsilon, learning_rate) {
  # Initialize a Q-matrix of size #states x #actions with zeroes
  Q_mat <- matrix(0, nrow=length(states), ncol=length(actions),
              dimnames=list(states, actions))
  # Run n iterations of Q-learning
  for (i in 1:n) {
    Q_mat <- updateIteration(initState, termState, epsilon, learning_rate, Q_mat)
  }
  return(Q_mat)
}

updateIteration <- function(initState, termState, epsilon, learning_rate, Q_mat) {
  state <- initState # set cursor to initial state
  while (state != termState) {
    # Select the next action greedily or randomnly
    if (runif(1) >= epsilon) {
      action <- sample(actions, 1) # Select randomnly
    } else {
      action <- which.max(Q_mat[state, ]) # Select best action
    }
    # Extract the next state and its reward
    response <- transitionStateAction(state, action)
    # Update the corresponding value in Q-matrix (learning)
    Q_mat[state, action] <- Q_mat[state, action] + learning_rate *
      (response$reward + max(Q_mat[response$state, ]) - Q_mat[state, action])
    state <- response$state # update with next state
  }
  return(Q_mat)
}

## Define Learning parameters
epsilon <- 0.1
learning_rate <- 0.9

## Perform Q Learning
Q_mat <- Qlearning(500000, "s1", "s15", epsilon, learning_rate)
Q_mat
actions[max.col(Q_mat)]

