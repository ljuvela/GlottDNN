#include <gslwrap/min_fminimizer.h>
#include <gslwrap/multimin_fdfminimizer.h>
#include <math.h>

struct my_f : public gsl::min_f
{
	double a;
	double b;
	double c;
	virtual double operator()(const double& x)
	{
		return (a*x+b)*x+c;
	}
};

struct fn1 : public gsl::min_f
{
	virtual double operator()(const double& x)
	{
		return cos(x)+1.0;
	}	
};

void OneDimMinimiserTest()
{
	fn1 f;
//	gsl::min_fminimizer m(gsl_min_fminimizer_goldensection);
	gsl::min_fminimizer m;
	double mm=2;
	double a=0;
	double b=6;
	m.set(f, mm, a, b);
	int status;
	do{
		status=m.iterate();
		mm=m.minimum();
		a=m.x_lower();
		b=m.x_upper();
		
		status =gsl_min_test_interval(a,b,0.001, 0.0);

		if (status==GSL_SUCCESS)
			printf("Converged:\n");

		printf("%5d [%.7f, %.7f] "
			   "%.7f %.7f %+.7f %.7f\n",
			   m.GetNIterations(), a, b, mm, M_PI, mm-M_PI, b-a);
		
	}while (status!=GSL_SUCCESS && !m.is_converged());
}

struct my_function : public gsl::multimin_fdf
{
	my_function() : multimin_fdf(2){;}

	double a;
	double b;
	
	virtual double operator()(const gsl::vector& x)
	{
		return 10.0*(x[0]-a)*(x[0]-a)+20.0*(x[1]-b)*(x[1]-b)+30;
	}

	virtual void derivative(const gsl::vector& x, gsl::vector& g)
	{
		g[0]=20.0*(x[0]-a);
		g[1]=40.0*(x[1]-b);
	}
};

/*
using namespace::std;

void MultDimMinimiserTest2()
{
	my_function f;
	f.a=1.0;
	f.b=2.0;
	gsl::vector x(2);
	x[0]=1;
	x[1]=2;
	cout << "Value at f(x) x=" << x << endl;
	cout << f(x) << endl;

	cout << "Gradient at x=" << x << endl;
	gsl::vector g(2);
	f.derivative(x, g);
	cout << g << endl;
}
*/

void MultDimMinimiserTest()
{
	my_function f;
	uint dim=2;
	f.a=1.0;
	f.b=2.0;
	gsl::multimin_fdfminimizer mm(dim);

	// starting point
	gsl::vector x(dim);
	x[0]=5.0;
	x[1]=7.0;

	mm.set(f, x, 0.01, 1e-4);
	int status;
	uint iter=0;
	do{
		iter++;
		status=mm.iterate();

		if (status)
			break;
		
		status = gsl_multimin_test_gradient(mm.gradient().gslobj(), 1e-3);

		if (status==GSL_SUCCESS)
			printf("Minimum found at: \n");

		printf("%5d %.5f %.5f %10.5f\n", iter, 
			   mm.x_value()[0], 
			   mm.x_value()[1], 
			   mm.minimum());
	}while(status==GSL_CONTINUE && iter < 100);
}
