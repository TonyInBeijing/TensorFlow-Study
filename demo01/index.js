/**
 * @description 使用 tf.js 预测房价和面积的关系 房价 y = 权重 w * 面积 x
 * @author YHW 2022-01-14
 */

const tf = require("@tensorflow/tfjs-node");

// 定义一个面积的数组
const xData = [60, 70, 90, 130, 150];
// 定义一个房价的数组
const yData = [200, 300, 400, 600, 800];

// 把基础数据转化为 tensor 对象方便输入到框架中

const xs = tf.tensor2d(xData, [xData.length, 1]);
const ys = tf.tensor2d(yData, [yData.length, 1]);

// 定义一个神经网络
const model = tf.sequential();
// 添加层
model.add(tf.layers.dense({ inputShape: [1], utils: 1 }));
model.add(tf.layers.batchNormalization());

// 定义损失函数，这里采用的是均方误差 MSE，可以类比这里是上面的误差计算
model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
});



// 开始训练网络
let trainLogs = [];
const result = model.fit(
    xs,
    yx,
    {
        epochs: 300,
        callbacks: {
            onEpochEnd: async (epochs, logs) => {
                console.log(logs, "???");
                trainLogs.push({
                    mse: Math.sqrt(logs.loss),
                });

            }
        }
    }
);