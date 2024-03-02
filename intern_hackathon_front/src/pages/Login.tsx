import logo from "../assets/logo.png";
const Login = () => {
  return (
    <div>
      <div>
        <img src={logo} alt="" />
        <h1 className="font-popins font-bold">Welcome back to GeneInsight!</h1>
        <p className="font-popins font-normal text-first-gray">
          Log in to continue your journey towards new discoveries
        </p>
        <div>
          <div>
            <h4 className="font-popins font-normal text-dark">Full Name</h4>
            <input
              type="text"
              className="border-[1px]  rounded-md py-3 focus:border-[1px] focus:border-primary"
            />
          </div>
          <div>
            <h4 className="">Email</h4>
            <input type="text" />
          </div>
          <div>
            <h4>Password</h4>
            <input type="password" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
