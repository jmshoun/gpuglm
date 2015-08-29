
#ifndef GLMFAMILY_H_
#define GLMFAMILY_H_

#include <string>

#include "linkFunctions.h"
#include "varianceFunctions.h"
#include "cumulantFunctions.h"

class glmFamily {

protected:
	linkFunction link;
	linkFunction invLink;
	linkFunction linkDerivative;
	varianceFunction variance;
	cumulantFunction cumulant;

	bool isCanonical;
	num_t scaleParameter;

public:
	// Constructors/Destructors ///////////////////////////////////////////////
	glmFamily(std::string linkName, std::string varianceName,
			bool _isCanonical, num_t _scaleParameter) {
		link = getLinkFunction(linkName);
		invLink = getInvLinkFunction(linkName);
		linkDerivative = getLinkDerivativeFunction(linkName);
		cumulant = getCumulantFunction(linkName);
		variance = getVarianceFunction(varianceName);

		isCanonical = _isCanonical;
		scaleParameter = _scaleParameter;

		return;
	}

	// Accessor Functions /////////////////////////////////////////////////////
	linkFunction getLink(void) const { return link; };
	linkFunction getInvLink(void) const { return invLink; };
	linkFunction getLinkDerivative(void) const { return linkDerivative; };
	varianceFunction getVariance(void) const { return variance; };
	cumulantFunction getCumulant(void) const { return cumulant; };
	bool getIsCanonical(void) const { return isCanonical; };
	num_t getScaleParameter(void) const { return scaleParameter; };
};

#endif /* GLMFAMILY_H_ */
