-- | Attempt to test the type system with linear regression

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TemplateHaskell, Rank2Types, NoMonomorphismRestriction, ExistentialQuantification #-}

import Control.Monad (replicateM, replicateM_)
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import qualified TensorFlow.Variable as TF

import Prelude hiding (id,(.))
import Control.Category
import Data.Complex (Complex)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Word (Word8, Word16, Word32, Word64)
-- import Numeric.Neural

class Param p where

instance (Param p, Param q) => Param(p,q)

type ParaLike p s a = (p,s) -> a 

data Para s a = forall p. Para(ParaLike p s a) 

comp :: Para s a -> Para a b -> Para s b
comp (Para f) (Para g) = Para $ \ ((p,q),s) -> g(q,f(p,s))

instance Category Para where
    id = Para $ \ ((),x) -> x 
    (Para g) . (Para f) = Para $ \ ((p,q),s) -> g(q,f(p,s))

-- Forward function of the model component
-- Simply constructs the predicted y values out of the given x, a and b
f_fwd :: forall v'1 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => TF.Tensor v'1 t -> (TF.Variable t, TF.Variable t) -> TF.Tensor TF.Build t
f_fwd x (a, b) = ( x `TF.mul` TF.readValue a) `TF.add` TF.readValue b

-- Reverse function of the model component
-- Meant to use the Transposed Jacobian of the forward function and multiply this by the loss values
-- Due to each equation being the same and containing only one variable, this simplifies to multiplying the loss values by a
f_rev :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => TF.Tensor v'1 t -> TF.Tensor v'2 t -> (TF.Variable t, TF.Variable t) -> (TF.Tensor TF.Build t, (TF.Variable t, TF.Variable t))
f_rev x loss (a, b) = (TF.readValue a `TF.mul` loss, (a, b))

-- Forward for the loss function
-- Simply takes the squared difference between the predicted (yhat) and true (y) values
loss_fwd :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        TF.Tensor v'1 t -> TF.Tensor v'2 t -> TF.Tensor TF.Build t
loss_fwd yHat y = TF.square (yHat `TF.sub` y)

-- Reverse for the loss function
-- Returns two nearly identical vectors containing the loss between predicted (yhat) and true (y) values, multiplied by the learning rate
loss_rev :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        TF.Tensor v'1 t -> TF.Tensor v'2 t -> TF.Variable t -> ( TF.Tensor TF.Build t, TF.Tensor TF.Build t)
loss_rev yHat y a = (TF.readValue a `TF.mul` (yHat `TF.sub` y), TF.readValue a `TF.mul` (y `TF.sub` yHat))

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- replicateM 100 randomIO
    let yData = [x*3 + 8 | x <- xData]
    -- Fit linear regression model.
    (w, b) <- fit xData yData
    assertBool "w == 3" (abs (3 - w) < 0.001)
    assertBool "b == 8" (abs (8 - b) < 0.001)

fit :: [Float] -> [Float] -> IO (Float, Float)
fit xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xData
        y = TF.vector yData
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable 0
    b <- TF.initializedVariable 0
    -- Define the loss function.
    let yHat = (x `TF.mul` TF.readValue w) `TF.add` TF.readValue b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (w', b')
