import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

#活性化関数
def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
  return (x > 0).astype(float)

def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x - c)
  sum_exp_x = np.sum(exp_x)
  return exp_x / sum_exp_x

#adamの更新
def adam_update(param, g, m, v, t, lr, b1, b2, eps):
  """
  param: 更新対象の重みやバイアス
  g: 勾配
  m, v: 過去の履歴配列
  t: ステップ
  lr: learning rate
  b1: beta1
  b2: beta2
  eps: 0を防ぐ
  """
  m[:] = b1 * m + (1.0 - b1) * g
  v[:] = b2 * v + (1.0 - b2) * (g ** 2)
  m_hat = m / (1.0 - b1 ** t)
  v_hat = v / (1.0 - b2 ** t)

  param -= lr * m_hat / (np.sqrt(v_hat) + eps)

if __name__ == "__main__":

  X_train = [] #入力データ
  Y_train = [] #出力データ

  #画像の読み込み
  for v in range(200):
    image_num = np.random.randint(10)
    image_name = f"./image/image_{image_num}"
    image_path = image_name + ".png"
    try:
      img = Image.open(image_path)
      img_grey = img.convert('L')
      img_resized = img_grey.resize((28, 28))
      img_2d = np.array(img_resized)
      img_1d = img_2d.flatten() 
      y = np.zeros(10)
      y[image_num] = 1
      X_train.append(img_1d)
      Y_train.append(y)
    except FileNotFoundError:
      print(f"{image_path}が見つからない\n")

  

  #adamで使用する
  beta1 = 0.9
  beta2 = 0.999
  learning_rate = 0.001
  eps = 10 ** -8

  input_size = 784
  hidden_size = 548
  output_size = 10 #数字0~9の識別

  weight_input_hidden = np.random.rand(hidden_size, input_size) * 0.01
  weight_hidden_output = np.random.rand(output_size, hidden_size) * 0.01

  bias_input_hidden = np.zeros(hidden_size)
  bias_hidden_output = np.zeros(output_size)

  #adamの初期化
  t_step = 0
  m_w_in_hid = np.zeros_like(weight_input_hidden)
  v_w_in_hid = np.zeros_like(weight_input_hidden)

  m_b_in_hid = np.zeros_like(bias_input_hidden)
  v_b_in_hid = np.zeros_like(bias_input_hidden)

  m_w_hid_out = np.zeros_like(weight_hidden_output)
  v_w_hid_out = np.zeros_like(weight_hidden_output)

  m_b_hid_out = np.zeros_like(bias_hidden_output)
  v_b_hid_out = np.zeros_like(bias_hidden_output)

  Ln = []

  for i in range(200):
    x_input = X_train[i]
    result = Y_train[i]

    #計算
    z_1 = x_input

    u_2 = weight_input_hidden @ z_1 + bias_input_hidden
    z_2 = relu(u_2)

    u_3 = weight_hidden_output @ z_2 + bias_hidden_output
    z_3 = softmax(u_3)

    loss = -np.sum(result * np.log(z_3 + 1e-7))
    Ln.append(loss)

    #誤差伝搬
    delta_3 = z_3 - result

    d_weight_hidden_output = np.outer(delta_3, z_2)
    d_bias_hidden_output = delta_3

    delta_2 = (weight_hidden_output.T @ delta_3) * relu_derivative(u_2)

    d_weight_input_hidden = np.outer(delta_2, x_input)
    d_bias_input_hidden = delta_2

    #重みの更新
    #hidden -> output
    t_step += 1
    adam_update(weight_hidden_output, d_weight_hidden_output, m_w_hid_out, v_w_hid_out, 
                t_step, learning_rate, beta1, beta2, eps)
    adam_update(bias_hidden_output, d_bias_hidden_output, m_b_hid_out, v_b_hid_out, 
                t_step, learning_rate, beta1, beta2, eps)
    #input -> hidden
    adam_update(weight_input_hidden, d_weight_input_hidden, m_w_in_hid, v_w_in_hid, 
                t_step, learning_rate, beta1, beta2, eps)
    adam_update(bias_input_hidden, d_bias_input_hidden, m_b_in_hid, v_b_in_hid, 
                t_step, learning_rate, beta1, beta2, eps)
    

  plt.figure(figsize=(10, 5))
  plt.plot(Ln, label="Raw Loss", color="blue", alpha=0.2)

  loss_smoothed = pd.Series(Ln).rolling(window=10).mean()
  plt.plot(loss_smoothed, color="red", linewidth=2, label="Smoothed Loss (MA 10)")

  plt.title("Loss History (Adam)")
  plt.xlabel("Steps")
  plt.ylabel("Loss")
  plt.grid(True)
  plt.legend()
  plt.show()
