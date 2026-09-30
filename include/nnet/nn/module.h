#ifndef MODULE_H
#define MODULE_H

class Module {
	public:
		virtual ~Module();
		
		virtual std::vector<Parameters*> parameters(){
			return {};
	};
	

};


#endif
