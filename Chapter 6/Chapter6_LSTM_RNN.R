# LSTM implementation
lstm<-function(x, weight, bias){
  # Unstack input into step_size
  x = tf$unstack(x, step_size, 1)
  
  # Define a lstm cell
  lstm_cell = tf$contrib$rnn$BasicLSTMCell(n.hidden, forget_bias=1.0, state_is_tuple=TRUE)
  
  # Get lstm cell output
  cell_output = tf$contrib$rnn$static_rnn(lstm_cell, x, dtype=tf$float32)
  
  # Linear activation, using rnn inner loop last output
  last_vec=tail(cell_output[[1]], n=1)[[1]]
  return(tf$matmul(last_vec, weights) + bias)
}