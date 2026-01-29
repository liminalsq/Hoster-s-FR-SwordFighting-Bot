-- HOSTER's FR Neural Network Module

local NeuralNetwork = {}
math.randomseed(os.time())

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

local function sigmoid_derivative(x)
    return x * (1 - x)
end

function NeuralNetwork:new()
    local obj = {}
    obj.w1 = {{math.random()*2-1, math.random()*2-1}, {math.random()*2-1, math.random()*2-1}}
    obj.w2 = {math.random()*2-1, math.random()*2-1}
    obj.b1 = {math.random()*2-1, math.random()*2-1}
    obj.b2 = math.random()*2-1
    obj.lr = 0.1
    setmetatable(obj, self)
    self.__index = self
    return obj
end

function NeuralNetwork:forward(x)
    self.h = {}
    for i = 1, 2 do
        self.h[i] = sigmoid(
            x[1]*self.w1[i][1] + x[2]*self.w1[i][2] + self.b1[i]
        )
    end

    self.o = self.h[1]*self.w2[1] + self.h[2]*self.w2[2] + self.b2
    self.o = sigmoid(self.o)
    return self.o
end

function NeuralNetwork:train(x, target)
    local output = self:forward(x)
    local error = target - output
    local d_output = error * sigmoid_derivative(output)

    for i = 1, 2 do
        self.w2[i] = self.w2[i] + self.lr * d_output * self.h[i]
    end
    self.b2 = self.b2 + self.lr * d_output

    for i = 1, 2 do
        local d_hidden = d_output * self.w2[i] * sigmoid_derivative(self.h[i])
        self.w1[i][1] = self.w1[i][1] + self.lr * d_hidden * x[1]
        self.w1[i][2] = self.w1[i][2] + self.lr * d_hidden * x[2]
        self.b1[i] = self.b1[i] + self.lr * d_hidden
    end
end

return NeuralNetwork
