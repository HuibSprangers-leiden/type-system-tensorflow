-- | Attempt to test the type system with linear regression

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TemplateHaskell, Rank2Types, NoMonomorphismRestriction, ExistentialQuantification #-}

import Control.Monad (replicateM, replicateM_)
import System.Random (randomIO)
import Test.HUnit ((@=?),assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF hiding (assignAdd)
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable,shape)
import qualified TensorFlow.Variable as TF
import qualified TensorFlow.Tensor as TF (toTensor)

import Prelude hiding (id,(.))
import Control.Category
import Control.Monad (zipWithM)
import Data.Complex (Complex)
import Data.Int (Int8, Int16, Int32, Int64)
import Data.Word (Word8, Word16, Word32, Word64)

-- import Text.XHtml (param)

import Control.Lens
-- import System.IO
-- import Control.Monad.State (State, execState, get)
-- import Control.Monad (when)

-- import Data.Set (Set, empty)
import Data.Either
-- import Data.Stream.Infinite (Stream(..))
import qualified Data.Vector as V
import qualified Control.Arrow as A

type D = (Int32,Int32)

addD :: D -> D -> D
addD d1 d2 = (fst d1 + fst d2, snd d1 + snd d2)

data T = E D D D
    | I D D
    deriving (Show,Eq)

-- To give the dimensions for the lens p a b serve as input parameters
data Term =
    Comp Term Term |
    PComp Term Term |
    Tensor Term Term |
    Hide Term |
    Id D |
    -- Loss and Alpha as base terms, see the type checker for their definitions
    Loss D D |
    Alpha |
    Model D D D |
    Grad D D |
    Ex1 D D D
    deriving (Show,Eq)

getInputD :: T -> D
getInputD (E _ a _) = a 
getInputD (I a _) = a

getParaD :: T -> D
getParaD (E p _ _) = p

getOutputD :: T -> D
getOutputD (E _ _ b) = b
getOutputD (I _ b) = b

infer :: Term -> Either String T
infer (Id a) = Right $ E (0,0) a a
infer (Loss p a) = Right $ E p a (1,1)
infer Alpha = Right $ E (0,0) (1,1) (0,0)
infer (Model p a b) = Right $ E p a b
infer (Grad p b) = Right $ E p (0,0) b
infer (Hide t) = do
    u <- infer t
    case u of
        (E p a b) -> Right $ I a b
        _ -> Left "Invalid input type for Hide"
infer (Comp t1 t2) = do
    u1 <- infer t1
    u2 <- infer t2
    case (u2, u1) of
        (E p1 a1 b1, E p2 a2 b2) ->
            if b1 == a2
                then Right $ E (addD p1 p2) a1 b2
            else Left "Unfitting Dimensions for Comp"
        (I a1 b1, I a2 b2) ->
            if b1 == a2
                then Right $ I a1 b2
            else Left "Unfitting Dimensions for Comp"
        _ -> Left "Invalid input type(s) for Comp"
infer (PComp t1 t2) = do
    u1 <- infer t1
    u2 <- infer t2
    case (u2, u1) of
        (E p1 a1 b1, E p2 a2 b2) ->
            if b1 == p2
                then Right $ E p1 a1 b2
            else Left "Unfitting Dimensions for PComp"
        _ -> Left "Invalid input type(s) for PComp"
infer (Tensor t1 t2) = do
    u1 <- infer t1
    u2 <- infer t2
    case (u1, u2) of
        (E p1 a1 b1, E p2 a2 b2) -> Right $ E (addD p1 p2) (addD a1 a2) (addD b1 b2)
        (I a1 b1, I a2 b2) -> Right $ I (addD a1 a2) (addD b1 b2)
        _ -> Left "Invalid input type(s) for Tensor"

interp :: forall t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => Term -> Lens (TF.Tensor TF.Build t,TF.Tensor TF.Build t) (TF.Tensor TF.Build t,TF.Tensor TF.Build t) (TF.Tensor TF.Build t) (TF.Tensor TF.Build t)
interp (Loss p a) = lens loss_fwd loss_rev
interp Alpha = lens alpha_fwd alpha_rev
interp (Model p a b) = lens f_fwd f_rev
interp (Grad p b) = lens grad_fwd grad_rev
interp (Comp t1 t2) = do
    let lens1 = interp t1
    let lens2 = interp t2

    let paramD1 = head (rights [fmap getParaD (infer t1)])
    let pd1 = TF.vector [(fst (paramD1))]
    let paramD2 = head (rights [fmap getParaD (infer t2)])
    let pd2 = TF.vector [(fst (paramD2))]

    let lens1_fwd = (^# lens1)
    let lens2_fwd = (^# lens2)
    let res_fwd = A.first (A.arr (\ (pq) -> (TF.slice pq 0 pd1, TF.slice pq pd1 pd2))) A.>>> A.arr (\ ((p,q),a) -> ((p,a),q)) A.>>> A.first lens1_fwd A.>>>  A.arr (\ (x,y) -> (y,x)) A.>>> A.arr (\ (x,y) -> lens2_fwd (x,y))

    let lens1_rev = (flip (storing lens1))
    let lens2_rev = (flip (storing lens2))
    let res_rev = A.first (A.first (A.arr (\ (pq) -> (TF.slice pq 0 pd1, TF.slice pq pd1 pd2)))) A.>>> A.arr (\ (((p,q),a),b) -> ((((p,a),q),b),(p,a))) A.>>> A.first (A.first (A.first lens1_fwd)) A.>>> A.first (A.arr (\ ((a,q),b) -> (lens2_rev (q,a) b))) A.>>> A.arr (\ ((p,a),(q,b)) -> ((lens1_rev (q,b) a),p)) A.>>> A.arr (\ ((p,a),q) -> ( TF.concat 0 [p, q], a))

    lens res_fwd (curry res_rev)
interp (Tensor t1 t2) = do
    let lens1 = interp t1
    let lens2 = interp t2

    let paramD1 = head (rights [fmap getParaD (infer t1)])
    let pd1 = TF.vector [(fst (paramD1))]
    let paramD2 = head (rights [fmap getParaD (infer t2)])
    let pd2 = TF.vector [(fst (paramD2))]
    let inputD1 = head (rights [fmap getInputD (infer t1)])
    let id1 = TF.vector [(fst (inputD1))]
    let inputD2 = head (rights [fmap getInputD (infer t2)])
    let id2 = TF.vector [(fst (inputD2))]
    let outputD1 = head (rights [fmap getOutputD (infer t1)])
    let od1 = TF.vector [(fst (outputD1))]
    let outputD2 = head (rights [fmap getOutputD (infer t2)])
    let od2 = TF.vector [(fst (outputD2))]

    let lens1_fwd = (^# lens1)
    let lens2_fwd = (^# lens2)
    let res_fwd = A.arr (\ (p,a) -> ((TF.slice p 0 pd1, TF.slice p pd1 pd2), (TF.slice a 0 id1, TF.slice a id1 id2))) A.>>> A.arr (\ ((p1,p2),(a1,a2)) -> ( TF.concat 1 [lens1_fwd (p1,a1), lens2_fwd (p2,a2)]))

    let lens1_rev = (flip (storing lens1))
    let lens2_rev = (flip (storing lens2))
    let res_rev = A.arr (\ ((p,a),b) -> (((TF.slice p 0 pd1, TF.slice p pd1 pd2), (TF.slice a 0 id1, TF.slice a id1 id2)), (TF.slice b 0 od1, TF.slice b od1 od2))) A.>>> A.arr (\ (((p1,p2),(a1,a2)),(b1,b2)) -> (lens1_rev (p1,a1) b1, lens2_rev (p2,a2) b2)) A.>>> A.arr (\ ((p1,a1),(p2,a2)) -> (TF.concat 0 [p1, p2], TF.concat 0 [a1, a2]))

    lens res_fwd (curry res_rev)
interp (PComp t1 t2) = do
    let lens1 = interp t1
    let lens2 = interp t2

    let inputD1 = head (rights [fmap getInputD (infer t1)])
    let id1 = TF.vector [(fst (inputD1))]
    let inputD2 = head (rights [fmap getInputD (infer t2)])
    let id2 = TF.vector [(fst (inputD2))]

    let lens1_fwd = (^# lens1)
    let lens2_fwd = (^# lens2)
    let res_fwd = A.arr (\ (p,a) -> (p, (TF.slice a 0 id1, TF.slice a id1 id2))) A.>>> A.arr (\ (p,(a1,a2)) -> (lens1_fwd (p,a1),a2)) A.>>> lens2_fwd

    let lens1_rev = (flip (storing lens1))
    let lens2_rev = (flip (storing lens2))
    let res_rev = A.arr (\ ((p,a),b) -> ((p, (TF.slice a 0 id1, TF.slice a id1 id2)),b)) A.>>> A.arr (\ ((p,(a1,a2)),b) -> (((lens1_fwd (p,a1),a2),b),(p,a1))) A.>>> A.first (A.arr (\ ((p,a),b) -> lens2_rev (p,a) b)) A.>>> A.arr (\ ((p1,a1),(p2,a2)) -> (lens1_rev (p2,a2) p1,a1)) A.>>> A.arr (\ ((p,a1),a2) -> (p,TF.concat 0 [a1, a2]))

    lens res_fwd (curry res_rev)

-- Forward function of the model component
-- Simply constructs the predicted y values out of the given x, a and b
f_fwd :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => (TF.Tensor v'1 t, TF.Tensor v'2 t) -> TF.Tensor TF.Build t
f_fwd (p, a) = ( a `TF.mul` (TF.slice p 0 (TF.constant (TF.Shape [1]) [1 :: Int32]))) `TF.add` (TF.slice p 1 (TF.constant (TF.Shape [1]) [1 :: Int32]))

-- Reverse function of the model component
-- Meant to use the Transposed Jacobian of the forward function and multiply this by the loss values
-- Due to each equation being the same and containing only one variable, this simplifies to multiplying the loss values by the first parameter
f_rev :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => (TF.Tensor v'1 t, TF.Tensor TF.Build t) -> TF.Tensor v'2 t -> (TF.Tensor TF.Build t, TF.Tensor TF.Build t)
f_rev (p, a) b = (TF.concat 0 [TF.mean (a `TF.mul` b) (TF.constant (TF.Shape [1]) [0 :: Int32]), TF.mean b (TF.constant (TF.Shape [1]) [0 :: Int32])], a)

-- Forward for the loss function
-- Simply takes the squared difference between the predicted (yhat) and true (y) values
loss_fwd :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        (TF.Tensor v'1 t, TF.Tensor v'2 t) -> TF.Tensor TF.Build t
loss_fwd (yHat, y) = TF.square (yHat `TF.sub` y)

-- Reverse for the loss function
-- Returns two nearly identical vectors containing the loss between predicted (yhat) and true (y) values, multiplied by the learning rate
loss_rev :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        (TF.Tensor v'1 t, TF.Tensor TF.Build t) -> TF.Tensor v'2 t -> ( TF.Tensor TF.Build t, TF.Tensor TF.Build t)
loss_rev (yHat, y) a = (y, a `TF.mul` (y `TF.sub` yHat))

alpha_fwd :: forall v'1 v'2 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        (TF.Tensor v'1 t, TF.Tensor v'2 t) -> TF.Tensor TF.Build t
alpha_fwd (_,_) = TF.constant (TF.Shape [0]) []

alpha_rev :: forall v'1 v'2 v'3 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        (TF.Tensor v'1 t, TF.Tensor v'2 t) -> TF.Tensor TF.Build t -> ( TF.Tensor TF.Build t, TF.Tensor TF.Build t)
alpha_rev (_,_) a = (TF.constant (TF.Shape [0]) [], a)

grad_fwd :: forall v'1 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        (TF.Tensor TF.Build t, TF.Tensor v'1 t) -> TF.Tensor TF.Build t
grad_fwd (p,_) = p

grad_rev :: forall v'1 v'2 v'3 t . (TF.OneOf '[(Data.Complex.Complex Double),
                                    (Data.Complex.Complex Float),
                                    Data.Int.Int16, Data.Int.Int32,
                                    Data.Int.Int64, Data.Int.Int8,
                                    Data.Word.Word16, Data.Word.Word8, Double,
                                    Float] t) => 
                        (TF.Tensor v'1 t, TF.Tensor v'2 t) -> TF.Tensor v'3 t -> ( TF.Tensor TF.Build t, TF.Tensor TF.Build t)
grad_rev (p,_) b = (p `TF.add` b, TF.constant (TF.Shape [0]) [])

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- replicateM 100 randomIO
    let yData = [x*3 + 8 | x <- xData]
    -- Fit linear regression model.
    (w, b) <- fit xData yData
    assertBool "w == 3" (abs (3 - w) < 0.001)
    assertBool "b == 8" (abs (8 - b) < 0.001)
    (w',b') <- typedFit xData yData
    assertBool "w' == 3" (abs (3 - w') < 0.001)
    assertBool "b' == 8" (abs (8 - b') < 0.001)

-- This is the function for the typed regression
typedFit :: [Float] -> [Float] -> IO (Float, Float)
typedFit xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xData
        y = TF.vector yData
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable 0
    b <- TF.initializedVariable 0
    -- Define the loss function.
    let yHat = (x `TF.mul` TF.readValue w) `TF.add` TF.readValue b
        loss = TF.square (yHat `TF.sub` y)
    
    -- let f_lens = interp (Model (2,2) (100,100) (100,100))
    -- let loss_lens = interp (Loss (100,100) (100,100))
    -- let alpha_lens = interp (Alpha)
    -- let grad_lens = interp (Grad (2,2) (2,2))
    let pipeline = interp (Comp (PComp (Grad (2,2) (2,2)) (Model (2,2) (100,100) (100,100))) (Comp (Loss (100,100) (100,100)) (Alpha)))
    -- Optimize with gradient descent.
    let mod_rev = set f_lens y (TF.concat 0 [TF.readValue w, TF.readValue b],x)
    let res_rev = set pipeline (TF.constant (TF.Shape [1]) [0.001] ) (TF.concat 0 [TF.concat 0 [TF.readValue w, TF.readValue b],y],x)
    
    let sq_param = TF.slice (fst mod_rev) (TF.constant (TF.Shape [1]) [0 :: Int32]) (TF.constant (TF.Shape [1]) [1 :: Int32])
    TF.group =<< zipWithM TF.assignAdd [w,b] [sq_param]
    -- trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
    -- replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (w', b')

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
    
    -- let f_lens = interp (Model (2,2) (100,100) (100,100))
    -- let loss_lens = interp (Loss (100,100) (100,100))
    -- let alpha_lens = interp (Alpha)
    -- let pipeline = interp (Comp (Model (2,2) (100,100) (100,100)) (Comp (Loss (100,100) (100,100)) (Alpha)))
    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (w', b')
