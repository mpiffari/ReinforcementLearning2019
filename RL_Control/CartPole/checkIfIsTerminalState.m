function isTerminal = checkIfIsTerminalState(state)
    global cart
    superiorAngle = cart.maximumAngle;
    inferiorAngle = - cart.maximumAngle;
    actualAngle = state(2);
    if (actualAngle > superiorAngle) || (actualAngle < inferiorAngle)
        isTerminal = 1;
        disp('POLE GOES OUT OF THE LIMIT')
    else
        isTerminal = 0;
    end
end

