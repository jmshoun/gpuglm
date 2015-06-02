
#ifndef GLMCONTROL_H_
#define GLMCONTROL_H_

#include <string>

class glmControl {

protected:
	std::string fitMethod;
	unsigned int maxIterations;
	double tolerance;

public:
	// Constructors/Destructors ///////////////////////////////////////////////
	glmControl(std::string _fitMethod, unsigned int _maxIterations,
			double _tolerance) {
		fitMethod = _fitMethod;
		maxIterations = _maxIterations;
		tolerance = _tolerance;
	}

	// Accessor Methods ///////////////////////////////////////////////////////
	std::string getFitMethod(void) { return fitMethod; };
	unsigned int getMaxIterations(void) { return maxIterations; };
	double getTolerance(void) { return tolerance; };
};

#endif /* GLMCONTROL_H_ */
