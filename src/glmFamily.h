
#ifndef GLMFAMILY_H_
#define GLMFAMILY_H_

#include <string>

#include "linkFunctions.h"
#include "varianceFunctions.h"

class glmFamily {

protected:
	linkFunction link;
	linkFunction invLink;
	varianceFunction variance;

public:
	// Constructors/Destructors ///////////////////////////////////////////////
	glmFamily(std::string linkName, std::string varianceName) {
		link = getLinkFunction(linkName);
		invLink = getInvLinkFunction(linkName);
		variance = getVarianceFunction(varianceName);

		return;
	}

	// Accessor Functions /////////////////////////////////////////////////////
	linkFunction getLink(void) { return link; };
	linkFunction getInvLink(void) { return invLink; };
	varianceFunction getVariance(void) { return variance; };
};

#endif /* GLMFAMILY_H_ */
