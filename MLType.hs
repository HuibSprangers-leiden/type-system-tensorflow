{-# LANGUAGE TemplateHaskell, Rank2Types, NoMonomorphismRestriction #-}

module MLType where
    
-- import Prelude hiding (id,(.))
import Control.Lens
-- import Control.Category
-- import Numeric.Neural

type D = (Int,Int)

addD :: D -> D -> D
addD d1 d2 = (fst d1 + fst d2, snd d1 + snd d2)

data T = E D D D 
    | I D D
    deriving (Show,Eq)

-- To give the dimensions for the lens p a b serve as input parameters
data Term = 
    -- For recursion the terms will need to eventually reach an Implicit lens equivalent to I D D or an Explicit lens equivalent to E D D D
    -- Exp D D D | 
    -- Imp D D | 
    Comp (Term) (Term) | 
    Tensor (Term) (Term) | 
    Hide (Term) | 
    -- Loss and Alpha as base terms, see the type checker for their definitions
    Loss D D | 
    Alpha
    deriving (Show,Eq)

typeCheck :: (Term) -> T -> Either String T
-- typeCheck (Exp p a b) = Right (E p a b)
-- typeCheck (Imp a b) = Right (I a b)
typeCheck (Loss p a) (E q b c) = 
    if p == q
        then do
            if a == b
                then do
                    if c == (1,1)
                        then Right (E p a (1,1))
                    else Left "Incorrect type for Loss"
            else Left "Incorrect type for Loss"
    else Left "Incorrect type for Loss"
typeCheck (Alpha) (E p a b) =
    if p == (0,0)
        then do
            if a == (1,1)
                then do
                    if b == (0,0)
                        then Right (E (0,0) (1,1) (0,0))
                    else Left "Incorrect type for Alpha"
            else Left "Incorrect type for Alpha"
    else Left "Incorrect type for Alpha"
typeCheck (Hide t) (E q c d)=
    case typeCheck t (E q c d) of
        -- Checks if the type is explicit, then returns an implicit version or an error message
        Right (E p a b) -> 
            if a == c
                then do
                    if b == d
                        then Right (I a b)
                    else Left "Incorrect type for Hide"
            else Left "Incorrect type for Hide"
        _ -> Left "Unsupported type for Hide"
-- typeCheck (Comp t1 t2) = 
--     case typeCheck t1 of
--         -- Based on whether the first type is explicit or implicit, attaches the components
--         Right (E p a b) ->
--             case typeCheck t2 of
--                 -- Implicit and Explicit lenses can be composed together, which results in an explicit lens
--                 Right (E q c d) -> 
--                     -- Checks whether the dimension of the composed lenses will match, if not return an error
--                     if b == c
--                         -- Dimensions of the parameters are added together for explicit lenses
--                         then Right (E (addD p q) a d)
--                     else Left "Unfitting Dimensions for Comp"
--                 Right (I c d) -> 
--                     if b == c
--                         then Right (E p a d)
--                     else Left "Unfitting Dimensions for Comp"
--                 _ -> Left "Types Mismatch or Unsupported type for Comp"
--         Right (I a b) ->
--             case typeCheck t2 of
--                 Right (I c d) -> 
--                     if b == c
--                         then Right (I a d)
--                     else Left "Unfitting Dimensions for Comp"
--                 Right (E q c d) -> 
--                     if b == c
--                         then Right (E q a d)
--                     else Left "Unfitting Dimensions for Comp"
--                 _ -> Left "Types Mismatch or Unsupported type for Comp"
--         _ -> Left "Unsupported type for Comp"
-- typeCheck (Tensor t1 t2) = 
--     case typeCheck t1 of
--         -- Tensor adds all dimensions within the equivalent types together
--         Right (E p a b) ->
--             case typeCheck t2 of
--                 Right (E q c d) -> Right (E (addD p q) (addD a c) (addD b d))
--                 -- Implicit and Explicit lenses can be tensored together, which results in an explicit lens
--                 Right (I c d) -> Right (E p (addD a c) (addD b d))
--                 _ -> Left "Types Mismatch or Unsupported type for Tensor"
--         Right (I a b) ->
--             case typeCheck t2 of
--                 Right (I c d) -> Right (I (addD a c) (addD b d))
--                 Right (E q c d) -> Right (E q (addD a c) (addD b d))
--                 _ -> Left "Types Mismatch or Unsupported type for Tensor"
--         _ -> Left "Unsupported type for Tensor"

-- comp :: Term -> Term -> Term
-- comp (Imp a b) (Imp c d) = Imp a d
-- comp (Exp p a b) (Exp q c d) = Exp (addD p q) a d

-- tensor :: Term p a b -> Term q c d -> Term r e f
-- te

-- paralike :: ((a,p) -> b) -> ((a,p) -> b -> (a,p)) -> Term p a b
-- paralike apb apbap afa sp = 
